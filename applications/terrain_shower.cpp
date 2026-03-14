/*
 * terrain_shower.cpp
 *
 * Air-shower simulation over a triangular terrain mesh with a rectangular
 * detector box as the primary observation surface.
 *
 * The geometry is defined in a local East-North-Up (ENU) frame centred on
 * the given latitude/longitude.  Two PLY meshes in ECEF coordinates drive
 * the simulation:
 *
 *   --box-ply      Closed rectangular box (required).  Produced by
 *                  make_box_scene.jl.  Particles crossing any face are
 *                  recorded and removed.
 *
 *   --terrain-ply  Surrounding terrain absorber (optional).  Produced by
 *                  terrain_h5_to_ply.jl.  Particles hitting the terrain
 *                  are recorded and removed.
 *
 * An ObservationPlane placed 10 m below the bottom of the box catches any
 * particles that miss both the box and the terrain mesh.
 *
 * Zenith and azimuth are given in the local ENU frame at (--lat, --lon).
 * Azimuth is measured clockwise from geographic North.
 */

/* clang-format off */
#include <corsika/framework/process/InteractionCounter.hpp>
/* clang-format on */

#include <corsika/framework/core/Cascade.hpp>
#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include <corsika/framework/geometry/Sphere.hpp>
#include <corsika/framework/geometry/TriangularMesh.hpp>
#include <corsika/framework/process/DynamicInteractionProcess.hpp>
#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/framework/process/SwitchProcessSequence.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/framework/random/PowerLawDistribution.hpp>
#include <corsika/framework/utility/CorsikaFenv.hpp>
#include <corsika/framework/utility/CorsikaData.hpp>
#include <corsika/framework/utility/SaveBoostHistogram.hpp>

#include <corsika/modules/writers/EnergyLossWriter.hpp>
#include <corsika/modules/writers/InteractionWriter.hpp>
#include <corsika/modules/writers/LongitudinalWriter.hpp>
#include <corsika/modules/writers/ProductionWriter.hpp>
#include <corsika/modules/writers/PrimaryWriter.hpp>
#include <corsika/modules/writers/SubWriter.hpp>
#include <corsika/modules/writers/ParticleWriterParquet.hpp>
#include <corsika/output/OutputManager.hpp>

#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/magnetic/GeomagneticModel.hpp>
#include <corsika/media/refractivity/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/media/density_and_composition/HomogeneousMedium.hpp>
#include <corsika/media/interfaces/IMagneticFieldModel.hpp>
#include <corsika/media/LayeredSphericalAtmosphereBuilder.hpp>
#include <corsika/media/medium/MediumPropertyModel.hpp>
#include <corsika/media/composition/NuclearComposition.hpp>
#include <corsika/media/ShowerAxis.hpp>
#include <corsika/media/magnetic/UniformMagneticField.hpp>

#include <corsika/modules/BetheBlochPDG.hpp>
#include <corsika/modules/Epos.hpp>
#include <corsika/modules/EposLhcr.hpp>
#include <corsika/modules/ObservationMesh.hpp>
#include <corsika/modules/ObservationPlane.hpp>
#include <corsika/modules/PROPOSAL.hpp>
#include <corsika/modules/ParticleCut.hpp>
#include <corsika/modules/Pythia8.hpp>
#include <corsika/modules/QGSJetII.hpp>
#include <corsika/modules/QGSJetIII.hpp>
#include <corsika/modules/Sibyll.hpp>
#include <corsika/modules/Sophia.hpp>
#include <corsika/modules/StackInspector.hpp>
#include <corsika/modules/thinning/EMThinning.hpp>
#include <corsika/modules/LongitudinalProfile.hpp>
#include <corsika/modules/ProductionProfile.hpp>

#ifdef WITH_FLUKA
#include <corsika/modules/FLUKA.hpp>
#else
#include <corsika/modules/UrQMD.hpp>
#endif

#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupTrajectory.hpp>
#include <corsika/setup/SetupC7trackedParticles.hpp>

#include <boost/filesystem.hpp>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>

using namespace corsika;
using namespace std;

using EnvironmentInterface = media::IRefractiveIndexModel<
    media::IMediumPropertyModel<media::IMagneticFieldModel<media::IMediumModel>>>;
using EnvType      = media::Environment<EnvironmentInterface>;
using StackType    = setup::Stack<EnvType>;
using TrackingType = setup::Tracking;
using Particle     = StackType::particle_type;

// ---------------------------------------------------------------------------
// Random stream registration
// ---------------------------------------------------------------------------
long registerRandomStreams(long seed) {
  RNGManager<>::getInstance().registerRandomStream("cascade");
  RNGManager<>::getInstance().registerRandomStream("qgsjet");
  RNGManager<>::getInstance().registerRandomStream("qgsjetIII");
  RNGManager<>::getInstance().registerRandomStream("sibyll");
  RNGManager<>::getInstance().registerRandomStream("sophia");
  RNGManager<>::getInstance().registerRandomStream("epos");
  RNGManager<>::getInstance().registerRandomStream("epos-lhcr");
  RNGManager<>::getInstance().registerRandomStream("pythia");
  RNGManager<>::getInstance().registerRandomStream("urqmd");
  RNGManager<>::getInstance().registerRandomStream("fluka");
  RNGManager<>::getInstance().registerRandomStream("proposal");
  RNGManager<>::getInstance().registerRandomStream("thinning");
  RNGManager<>::getInstance().registerRandomStream("primary_particle");
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
    CORSIKA_LOG_INFO("random seed (auto) {}", seed);
  } else {
    CORSIKA_LOG_INFO("random seed {}", seed);
  }
  RNGManager<>::getInstance().setSeed(seed);
  return seed;
}

template <typename T>
using MyExtraEnv = media::GladstoneDaleRefractiveIndex<
    media::MediumPropertyModel<media::UniformMagneticField<T>>>;

// ---------------------------------------------------------------------------
// Build the local ENU basis vectors at (lat_deg, lon_deg).
// east, north, up are unit vectors expressed in the ECEF frame.
// ---------------------------------------------------------------------------
static void latLonToENU(double lat_deg, double lon_deg,
                        std::array<double, 3>& east,
                        std::array<double, 3>& north,
                        std::array<double, 3>& up) {
  double const lat = lat_deg * M_PI / 180.0;
  double const lon = lon_deg * M_PI / 180.0;
  east  = {-std::sin(lon),  std::cos(lon),  0.0};
  north = {-std::sin(lat) * std::cos(lon),
           -std::sin(lat) * std::sin(lon),
            std::cos(lat)};
  up    = { std::cos(lat) * std::cos(lon),
            std::cos(lat) * std::sin(lon),
            std::sin(lat)};
}

// ---------------------------------------------------------------------------
// Area-weighted centroid of a TriangularMesh in ECEF metres.
// ---------------------------------------------------------------------------
static void computeMeshCentroid(TriangularMesh const& mesh,
                                CoordinateSystemPtr rootCS,
                                double& cx, double& cy, double& cz) {
  double totalArea = 0.0;
  cx = cy = cz = 0.0;
  for (size_t i = 0; i < mesh.getTriangleCount(); ++i) {
    Triangle const& tri = mesh.getTriangle(i);
    auto const& idx = tri.getVertexIndices();
    auto const c0 = mesh.getVertex(idx[0]).getCoordinates(rootCS);
    auto const c1 = mesh.getVertex(idx[1]).getCoordinates(rootCS);
    auto const c2 = mesh.getVertex(idx[2]).getCoordinates(rootCS);
    double const x0 = c0.getX() / 1_m, y0 = c0.getY() / 1_m, z0 = c0.getZ() / 1_m;
    double const x1 = c1.getX() / 1_m, y1 = c1.getY() / 1_m, z1 = c1.getZ() / 1_m;
    double const x2 = c2.getX() / 1_m, y2 = c2.getY() / 1_m, z2 = c2.getZ() / 1_m;
    double const ex = x1-x0, ey = y1-y0, ez = z1-z0;
    double const fx = x2-x0, fy = y2-y0, fz = z2-z0;
    double const area = 0.5 * std::sqrt((ey*fz - ez*fy)*(ey*fz - ez*fy) +
                                         (ez*fx - ex*fz)*(ez*fx - ex*fz) +
                                         (ex*fy - ey*fx)*(ex*fy - ey*fx));
    cx += area * (x0 + x1 + x2) / 3.0;
    cy += area * (y0 + y1 + y2) / 3.0;
    cz += area * (z0 + z1 + z2) / 3.0;
    totalArea += area;
  }
  if (totalArea > 0.0) { cx /= totalArea; cy /= totalArea; cz /= totalArea; }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {

  CLI::App app{"Air-shower simulation over terrain + rectangular detector box."};

  // ---- Primary ----
  int A = 0, Z = 0, nevent = 0;
  std::vector<double> cli_energy_range;

  auto opt_Z = app.add_option("-Z", Z, "Atomic number for primary")
                   ->check(CLI::Range(0, 26))->group("Primary");
  auto opt_A = app.add_option("-A", A, "Atomic mass number for primary")
                   ->needs(opt_Z)->check(CLI::Range(1, 58))->group("Primary");
  app.add_option("-p,--pdg", "PDG code (p=2212, gamma=22, e-=11, mu-=13)")
      ->excludes(opt_A)->excludes(opt_Z)->group("Primary");
  app.add_option("-E,--energy", "Primary energy in GeV")->default_val(0);
  app.add_option("--energy_range", cli_energy_range,
                 "Low and high values for primary energy range in GeV")
      ->expected(2)->check(CLI::PositiveNumber)->group("Primary");
  app.add_option("--eslope", "Spectral index for energy sampling, dN/dE = E^eslope")
      ->default_val(-1.0)->group("Primary");
  app.add_option("-z,--zenith",
                 "Shower zenith angle in local ENU frame (deg from vertical)")
      ->default_val(0.)->check(CLI::Range(0., 90.))->group("Primary");
  app.add_option("-a,--azimuth",
                 "Shower azimuth clockwise from North in local ENU frame (deg)")
      ->default_val(0.)->check(CLI::Range(0., 360.))->group("Primary");

  // ---- Geometry ----
  app.add_option("--box-ply", "Path to detector box PLY file (ECEF metres)")
      ->required()->check(CLI::ExistingFile)->group("Geometry");
  app.add_option("--terrain-ply",
                 "Path to terrain mesh PLY file (ECEF metres). "
                 "Leave empty to disable terrain absorber.")
      ->default_val("")->group("Geometry");
  app.add_option("--lat", "Geodetic latitude of box centre (degrees, -90..90)")
      ->required()->check(CLI::Range(-90., 90.))->group("Geometry");
  app.add_option("--lon", "Geodetic longitude of box centre (degrees, -180..180)")
      ->required()->check(CLI::Range(-180., 180.))->group("Geometry");
  app.add_option("--injection-altitude",
                 "Altitude above Earth surface of the injection sphere (m)")
      ->default_val(112.75e3)->check(CLI::PositiveNumber)->group("Geometry");

  // ---- Energy cuts ----
  app.add_option("--emcut",
                 "Min. kin. energy of photons/electrons/positrons (GeV)")
      ->default_val(0.01)->check(CLI::Range(1e-6, 1e13))->group("Config");
  app.add_option("--hadcut", "Min. kin. energy of hadrons (GeV)")
      ->default_val(0.02)->check(CLI::Range(0.02, 1e13))->group("Config");
  app.add_option("--mucut", "Min. kin. energy of muons (GeV)")
      ->default_val(0.01)->check(CLI::Range(1e-6, 1e13))->group("Config");
  app.add_option("--taucut", "Min. kin. energy of tau leptons (GeV)")
      ->default_val(0.01)->check(CLI::Range(1e-6, 1e13))->group("Config");
  app.add_option("--max-deflection-angle",
                 "Maximum deflection angle in tracking (radians)")
      ->default_val(0.2)->check(CLI::Range(1e-8, 1.))->group("Config");
  bool track_neutrinos = false;
  app.add_flag("--track-neutrinos", track_neutrinos, "Enable neutrino tracking")
      ->group("Config");
  bool track_charm = false;
  app.add_flag("--track-charm", track_charm,
               "Enable charmed hadron tracking")->group("Config");

  // ---- Hadronic model ----
  app.add_option("-M,--hadronModel", "High-energy hadronic model")
      ->default_val("SIBYLL-2.3d")
      ->check(CLI::IsMember({"SIBYLL-2.3d", "QGSJet-II.04", "QGSJet-III",
                             "EPOS-LHC", "EPOS-LHC-R", "Pythia8"}))
      ->group("Config");
  app.add_option("-T,--hadronModelTransitionEnergy",
                 "Transition energy between high-/low-energy hadronic models (GeV)")
      ->default_val(std::pow(10, 1.9))->check(CLI::NonNegativeNumber)->group("Config");

  // ---- Thinning ----
  app.add_option("--emthin", "EM thinning fraction of primary energy")
      ->default_val(1e-6)->check(CLI::Range(0., 1.))->group("Thinning");
  app.add_option("--max-weight",
                 "Max weight for EM thinning (0 = Kobal optimum * 0.5)")
      ->default_val(0)->check(CLI::NonNegativeNumber)->group("Thinning");
  bool multithin = false;
  app.add_flag("--multithin", multithin, "Keep thinned particles (weight=0)")
      ->group("Thinning");

  // ---- Output / misc ----
  app.add_option("-N,--nevent", nevent, "Number of showers to simulate")
      ->default_val(1)->check(CLI::PositiveNumber)->group("Output");
  app.add_option("-f,--filename", "Output directory")
      ->required()->check(CLI::NonexistentPath)->group("Output");
  bool compressOutput = false;
  app.add_flag("--compress", compressOutput, "Compress output to tarball")
      ->group("Output");
  app.add_option("-s,--seed", "Random number seed (0 = auto)")
      ->default_val(0)->check(CLI::NonNegativeNumber)->group("Misc");
  bool force_interaction = false;
  app.add_flag("--force-interaction", force_interaction,
               "Force first interaction at injection point")->group("Misc");
  bool force_decay = false;
  app.add_flag("--force-decay", force_decay, "Force primary to immediately decay")
      ->group("Misc");
  bool disable_interaction_hists = false;
  app.add_flag("--disable-interaction-histograms", disable_interaction_hists,
               "Disable saving interaction histograms")->group("Misc");
  app.add_option("-v,--verbosity", "Verbosity: warn, info, debug, trace")
      ->default_val("info")
      ->check(CLI::IsMember({"warn", "info", "debug", "trace"}))
      ->group("Misc");

  CLI11_PARSE(app, argc, argv);

  // ---- Verbosity ----
  if (app.count("--verbosity")) {
    auto const lv = app["--verbosity"]->as<std::string>();
    if (lv == "warn") logging::set_level(logging::level::warn);
    else if (lv == "info") logging::set_level(logging::level::info);
    else if (lv == "debug") logging::set_level(logging::level::debug);
    else if (lv == "trace") {
#ifndef _C8_DEBUG_
      CORSIKA_LOG_ERROR("trace log level requires a Debug build.");
      return 1;
#endif
      logging::set_level(logging::level::trace);
    }
  }

  // ---- Validate primary ID ----
  if (app.count("--pdg") == 0) {
    if (app.count("-A") == 0 || app.count("-Z") == 0) {
      CORSIKA_LOG_ERROR("If --pdg is not given, both -A and -Z are required.");
      return 1;
    }
  }

  // ---- Random streams ----
  auto const seed = registerRandomStreams(app["--seed"]->as<long>());

  /* === ENVIRONMENT === */
  EnvType env;
  CoordinateSystemPtr const& rootCS = env.getCoordinateSystem();
  Point const earthCenter{rootCS, 0_m, 0_m, 0_m};
  Point const earthSurface{rootCS, 0_m, 0_m, constants::EarthRadius::Mean};

  /* === ENU FRAME at (--lat, --lon) === */
  double const lat_deg = app["--lat"]->as<double>();
  double const lon_deg = app["--lon"]->as<double>();
  std::array<double, 3> eastHat, northHat, upHat;
  latLonToENU(lat_deg, lon_deg, eastHat, northHat, upHat);
  CORSIKA_LOG_INFO("Site: lat={:.4f} deg, lon={:.4f} deg", lat_deg, lon_deg);
  CORSIKA_LOG_INFO("Up (ECEF): ({:.4f}, {:.4f}, {:.4f})",
                   upHat[0], upHat[1], upHat[2]);

  /* === LOAD MESHES === */
  std::string const boxPlyPath = app["--box-ply"]->as<std::string>();
  CORSIKA_LOG_INFO("Loading box mesh: {}", boxPlyPath);
  TriangularMesh boxMesh = TriangularMesh::fromPLY(boxPlyPath, rootCS, 1_m, 1e-6_m);
  CORSIKA_LOG_INFO("Box mesh: {} vertices, {} triangles",
                   boxMesh.getVertexCount(), boxMesh.getTriangleCount());

  std::string const terrainPlyPath = app["--terrain-ply"]->as<std::string>();
  bool const useTerrainMesh =
      !terrainPlyPath.empty() && boost::filesystem::exists(terrainPlyPath);
  std::unique_ptr<TriangularMesh> terrainMeshPtr;
  if (useTerrainMesh) {
    CORSIKA_LOG_INFO("Loading terrain mesh: {}", terrainPlyPath);
    terrainMeshPtr = std::make_unique<TriangularMesh>(
        TriangularMesh::fromPLY(terrainPlyPath, rootCS, 1_m, 1e-6_m));
    CORSIKA_LOG_INFO("Terrain mesh: {} vertices, {} triangles",
                     terrainMeshPtr->getVertexCount(),
                     terrainMeshPtr->getTriangleCount());
  } else if (!terrainPlyPath.empty()) {
    CORSIKA_LOG_WARN("Terrain mesh not found: {} -- terrain absorber disabled",
                     terrainPlyPath);
  }

  /* === BOX CENTROID (shower core) === */
  double cx, cy, cz;
  computeMeshCentroid(boxMesh, rootCS, cx, cy, cz);
  CORSIKA_LOG_INFO("Box centroid (ECEF m): ({:.1f}, {:.1f}, {:.1f})", cx, cy, cz);
  double const centroidAlt =
      std::sqrt(cx*cx + cy*cy + cz*cz) - constants::EarthRadius::Mean / 1_m;
  CORSIKA_LOG_INFO("Box centroid altitude: {:.1f} m", centroidAlt);

  /* === CATCH PLANE: 10 m below the box bottom ===
   *
   * "Below" is measured along the local up direction at (lat, lon).
   * We project all box vertices onto upHat and find the minimum.
   * The catch plane is placed 10 m below that, perpendicular to upHat.
   */
  double minBoxProj = std::numeric_limits<double>::infinity();
  for (size_t vi = 0; vi < boxMesh.getVertexCount(); ++vi) {
    auto const coords = boxMesh.getVertex(vi).getCoordinates(rootCS);
    double const proj = coords.getX() / 1_m * upHat[0]
                      + coords.getY() / 1_m * upHat[1]
                      + coords.getZ() / 1_m * upHat[2];
    minBoxProj = std::min(minBoxProj, proj);
  }
  double const catchPlaneDist = minBoxProj - 10.0; // 10 m below box bottom
  Point const catchPlaneCenter{rootCS,
      catchPlaneDist * upHat[0] * 1_m,
      catchPlaneDist * upHat[1] * 1_m,
      catchPlaneDist * upHat[2] * 1_m};
  DirectionVector const catchNormal{rootCS, {upHat[0], upHat[1], upHat[2]}};
  DirectionVector const catchRefDir{rootCS, {eastHat[0], eastHat[1], eastHat[2]}};
  Plane const catchPlane{catchPlaneCenter, catchNormal};
  CORSIKA_LOG_INFO("Catch plane: {:.1f} m below box bottom  "
                   "(dist from Earth centre: {:.1f} m)", 10.0, catchPlaneDist);

  /* === ATMOSPHERE === */
  media::GeomagneticModel wmm(earthCenter, corsika_data("GeoMag/WMM.COF"));
  // Magnetic field from WMM ENU components at the target site.
  // Defaults to a mid-latitude approximation; override if needed.
  constexpr double B_E_T =  0.0;      // east  (T)
  constexpr double B_N_T =  20.0e-6;  // north (T)  ~typical horizontal
  constexpr double B_U_T =  40.0e-6;  // up    (T)  ~typical vertical
  double const Bx = B_E_T*eastHat[0] + B_N_T*northHat[0] + B_U_T*upHat[0];
  double const By = B_E_T*eastHat[1] + B_N_T*northHat[1] + B_U_T*upHat[1];
  double const Bz = B_E_T*eastHat[2] + B_N_T*northHat[2] + B_U_T*upHat[2];
  MagneticFieldVector const siteField{rootCS, Bx*1_T, By*1_T, Bz*1_T};

  media::create_5layer_atmosphere<EnvironmentInterface, MyExtraEnv>(
      env, media::AtmosphereId::USStdBK, earthCenter, 1.000327, earthSurface,
      media::Medium::AirDry1Atm, siteField);

  /* === PRIMARY PARTICLE ID === */
  Code beamCode;
  if (app.count("--pdg") > 0) {
    beamCode = convert_from_PDG(PDGCode(app["--pdg"]->as<int>()));
  } else {
    if ((A == 1) && (Z == 1)) beamCode = Code::Proton;
    else if ((A == 1) && (Z == 0)) beamCode = Code::Neutron;
    else beamCode = get_nucleus_code(A, Z);
  }

  HEPEnergyType eMin = 0_GeV, eMax = 0_GeV;
  if (app["--energy"]->as<double>() > 0.0) {
    eMin = eMax = app["--energy"]->as<double>() * 1_GeV;
  } else if (!cli_energy_range.empty()) {
    eMin = std::min(cli_energy_range[0], cli_energy_range[1]) * 1_GeV;
    eMax = std::max(cli_energy_range[0], cli_energy_range[1]) * 1_GeV;
  } else {
    CORSIKA_LOG_CRITICAL("Must set either --energy or --energy_range.");
    return 1;
  }

  /* === SHOWER DIRECTION === */
  double const thetaRad = app["--zenith"]->as<double>()  * M_PI / 180.0;
  double const phiRad   = app["--azimuth"]->as<double>() * M_PI / 180.0;

  // Propagation direction in ENU (downgoing, azimuth CW from North)
  double const dE = std::sin(thetaRad) * std::sin(phiRad);
  double const dN = std::sin(thetaRad) * std::cos(phiRad);
  double const dU = -std::cos(thetaRad);

  // Rotate to ECEF
  double const pnx = dE*eastHat[0] + dN*northHat[0] + dU*upHat[0];
  double const pny = dE*eastHat[1] + dN*northHat[1] + dU*upHat[1];
  double const pnz = dE*eastHat[2] + dN*northHat[2] + dU*upHat[2];
  DirectionVector const propDir{rootCS, {pnx, pny, pnz}};

  /* === INJECTION POINT ===
   *
   * The shower core is at the box centroid.  The injection point is found by
   * tracing the (reversed) shower direction upstream until it intersects a
   * sphere at the requested injection altitude above Earth's surface.
   */
  Point const showerCore{rootCS, cx*1_m, cy*1_m, cz*1_m};
  double const injAlt = app["--injection-altitude"]->as<double>();
  double const R_inj  = constants::EarthRadius::Mean / 1_m + injAlt;
  double const dot_cu = cx*(-pnx) + cy*(-pny) + cz*(-pnz);
  double const centroidR2 = cx*cx + cy*cy + cz*cz;
  double const disc = dot_cu*dot_cu - (centroidR2 - R_inj*R_inj);
  if (disc <= 0.0) {
    CORSIKA_LOG_CRITICAL(
        "Upstream ray from box centroid does not intersect injection altitude "
        "sphere (alt={:.1f} km). Check zenith angle and altitude.", injAlt / 1e3);
    return EXIT_FAILURE;
  }
  double const injDist = -dot_cu + std::sqrt(disc);
  Point const injectionPos =
      showerCore + DirectionVector{rootCS, {-pnx, -pny, -pnz}} * (injDist * 1_m);

  media::ShowerAxis const showerAxis{injectionPos,
                                     (showerCore - injectionPos) * 1.2, env};
  auto const dX = 10_g / square(1_cm);

  CORSIKA_LOG_INFO("Shower core (ECEF m): ({:.1f}, {:.1f}, {:.1f})", cx, cy, cz);
  CORSIKA_LOG_INFO("Injection distance: {:.1f} km", injDist / 1e3);
  CORSIKA_LOG_INFO("Propagation direction (ECEF): ({:.4f}, {:.4f}, {:.4f})",
                   pnx, pny, pnz);

  /* === OUTPUT MANAGER === */
  std::stringstream argsStr;
  for (int i = 0; i < argc; ++i) argsStr << argv[i] << " ";
  std::string const outFilename = app["--filename"]->as<std::string>();
  OutputManager output(outFilename, seed, argsStr.str(), compressOutput);

  EnergyLossWriter dEdX{showerAxis, dX};
  output.add("energyloss", dEdX);

  /* === PHYSICS PROCESSES === */
  DynamicInteractionProcess<StackType> heModel;
  set<Code> const trackedParticles =
      (track_charm ? corsika::setup::C7trackedParticlesAndCharm
                   : corsika::setup::C7trackedParticles);
  auto const all_elements = corsika::media::get_all_elements_in_universe(env);
  auto sibyll = std::make_shared<corsika::sibyll::Interaction>(
      all_elements, trackedParticles);

  if (auto const ms = app["--hadronModel"]->as<std::string>(); ms == "SIBYLL-2.3d") {
    heModel = DynamicInteractionProcess<StackType>{sibyll};
  } else if (ms == "QGSJet-II.04") {
    heModel = DynamicInteractionProcess<StackType>{
        std::make_shared<corsika::qgsjetII::Interaction>()};
  } else if (ms == "QGSJet-III") {
    heModel = DynamicInteractionProcess<StackType>{
        std::make_shared<corsika::qgsjetIII::Interaction>()};
  } else if (ms == "EPOS-LHC") {
    heModel = DynamicInteractionProcess<StackType>{
        std::make_shared<corsika::epos::Interaction>(trackedParticles)};
  } else if (ms == "EPOS-LHC-R") {
    heModel = DynamicInteractionProcess<StackType>{
        std::make_shared<corsika::EPOS_LHCR::Interaction>(trackedParticles)};
  } else if (ms == "Pythia8") {
    heModel = DynamicInteractionProcess<StackType>{
        std::make_shared<corsika::pythia8::Interaction>(trackedParticles)};
  } else {
    CORSIKA_LOG_CRITICAL("Invalid hadron model: {}", ms);
    return EXIT_FAILURE;
  }

  InteractionCounter heCounted{heModel};

  corsika::pythia8::Decay decaySequence;
  corsika::sophia::InteractionModel sophia;

  HEPEnergyType const emcut  = 1_GeV * app["--emcut"]->as<double>();
  HEPEnergyType const hadcut = 1_GeV * app["--hadcut"]->as<double>();
  HEPEnergyType const mucut  = 1_GeV * app["--mucut"]->as<double>();
  HEPEnergyType const taucut = 1_GeV * app["--taucut"]->as<double>();
  ParticleCut<SubWriter<decltype(dEdX)>> cut(emcut, emcut, hadcut, mucut, taucut,
                                             !track_neutrinos, dEdX);

  auto const prod_threshold = std::min({emcut, hadcut, mucut, taucut});
  set_energy_production_threshold(Code::Electron,  prod_threshold);
  set_energy_production_threshold(Code::Positron,  prod_threshold);
  set_energy_production_threshold(Code::Photon,    prod_threshold);
  set_energy_production_threshold(Code::MuMinus,   prod_threshold);
  set_energy_production_threshold(Code::MuPlus,    prod_threshold);
  set_energy_production_threshold(Code::TauMinus,  prod_threshold);
  set_energy_production_threshold(Code::TauPlus,   prod_threshold);

  HEPEnergyType const heThreshold =
      1_GeV * app["--hadronModelTransitionEnergy"]->as<double>();
  corsika::proposal::Interaction emCascade(
      env, sophia, sibyll->getHadronInteractionModel(), heThreshold);
  corsika::proposal::ContinuousProcess<SubWriter<decltype(dEdX)>>
      emContinuousProposal(env, dEdX);
  BetheBlochPDG<SubWriter<decltype(dEdX)>> emContinuousBethe{dEdX};
  struct EMHadronSwitch {
    bool operator()(Particle const& p) const { return is_hadron(p.getPID()); }
  };
  auto emContinuous =
      make_select(EMHadronSwitch(), emContinuousBethe, emContinuousProposal);

  LongitudinalWriter profile{showerAxis, dX};
  output.add("profile", profile);
  LongitudinalProfile<SubWriter<decltype(profile)>> longprof{profile};

  ProductionWriter prod_profile{showerAxis, dX};
  output.add("production_profile", prod_profile);
  ProductionProfile<SubWriter<decltype(prod_profile)>> prodprof{prod_profile};

#ifdef WITH_FLUKA
  corsika::fluka::Interaction leIntModel{all_elements};
#else
  corsika::urqmd::UrQMD leIntModel{};
#endif
  InteractionCounter leIntCounted{leIntModel};

  struct EnergySwitch {
    HEPEnergyType cutE_;
    EnergySwitch(HEPEnergyType e) : cutE_(e) {}
    bool operator()(Particle const& p) const {
      return p.getKineticEnergy() < cutE_;
    }
  };
  auto hadronSequence =
      make_select(EnergySwitch(heThreshold), leIntCounted, heCounted);

  /* === OBSERVATION MESHES === */
  ObservationMesh<TrackingType, ParticleWriterParquet> boxObs{
      boxMesh, true, 1e-6_m};
  output.add("particles", boxObs);

  /* === CATCH PLANE (absorbing: records particles that miss the box) === */
  ObservationPlane<TrackingType, ParticleWriterParquet> catchLevel{
      catchPlane, catchRefDir, true, 1e-6_m};
  output.add("catch", catchLevel);

  PrimaryWriter<TrackingType, ParticleWriterParquet> primaryWriter(boxMesh);
  output.add("primary", primaryWriter);

  InteractionWriter<TrackingType, ParticleWriterParquet>
      inter_writer(showerAxis, boxMesh);
  output.add("interactions", inter_writer);

  /* === SHOWER LOOP === */
  double const emthinfrac   = app["--emthin"]->as<double>();
  double const maxWeightArg = app["--max-weight"]->as<double>();
  double const eSlope       = app["--eslope"]->as<double>();
  double const maxDefl      = app["--max-deflection-angle"]->as<double>();
  int    const nev          = app["--nevent"]->as<int>();

  auto runOneShower = [&](auto& sequence, int i_shower,
                          HEPEnergyType primaryTotalEnergy) {
    HEPEnergyType const eKin = primaryTotalEnergy - get_mass(beamCode);
    double const maxW = (maxWeightArg > 0)
                            ? maxWeightArg
                            : 0.5 * emthinfrac * primaryTotalEnergy / 1_GeV;
    EMThinning thinning{emthinfrac * primaryTotalEnergy, maxW, !multithin};
    StackInspector<StackType> stackInspect(10000, false, primaryTotalEnergy);

    auto fullSequence = make_sequence(stackInspect, hadronSequence, decaySequence,
                                      emCascade, prodprof, emContinuous,
                                      longprof, sequence, inter_writer,
                                      thinning, cut);

    TrackingType tracking(maxDefl);
    StackType stack;
    Cascade EAS(env, tracking, fullSequence, output, stack);
    stack.clear();

    CORSIKA_LOG_INFO("Shower {} of {}  |  {} E_kin = {:.3e} GeV  "
                     "zen={:.1f} deg az={:.1f} deg",
                     i_shower, nev, beamCode, eKin / 1_GeV,
                     app["--zenith"]->as<double>(),
                     app["--azimuth"]->as<double>());

    auto const primaryProperties =
        std::make_tuple(beamCode, eKin, propDir.normalized(), injectionPos, 0_ns);
    stack.addParticle(primaryProperties);

    if (force_interaction) {
      CORSIKA_LOG_INFO("Forcing first interaction at injection point.");
      EAS.forceInteraction();
    }
    if (force_decay) {
      CORSIKA_LOG_INFO("Forcing primary decay.");
      EAS.forceDecay();
    }

    primaryWriter.recordPrimary(primaryProperties);
    EAS.run();

    if (!disable_interaction_hists) {
      auto const hists = heCounted.getHistogram() + leIntCounted.getHistogram();
      std::string const histDir = outFilename + "/interaction_hist";
      boost::filesystem::create_directories(histDir);
      save_hist(hists.labHist(),
                histDir + "/inthist_lab_" + std::to_string(i_shower) + ".npz", true);
      save_hist(hists.CMSHist(),
                histDir + "/inthist_cms_" + std::to_string(i_shower) + ".npz", true);
    }
  };

  if (useTerrainMesh) {
    // Terrain is purely absorbing — particles hitting the ground are removed
    // without being recorded.
    ObservationMesh<TrackingType, WriterOff> terrainAbs{
        *terrainMeshPtr, true, 1e-6_m};

    auto obsSequence = make_sequence(boxObs, terrainAbs, catchLevel);
    output.startOfLibrary();
    for (int i = 1; i <= nev; ++i) {
      PowerLawDistribution<HEPEnergyType> plRng(eSlope, eMin, eMax);
      HEPEnergyType const E = (eMax == eMin)
          ? eMin
          : plRng(RNGManager<>::getInstance().getRandomStream("primary_particle"));
      runOneShower(obsSequence, i, E);
    }
  } else {
    auto obsSequence = make_sequence(boxObs, catchLevel);
    output.startOfLibrary();
    for (int i = 1; i <= nev; ++i) {
      PowerLawDistribution<HEPEnergyType> plRng(eSlope, eMin, eMax);
      HEPEnergyType const E = (eMax == eMin)
          ? eMin
          : plRng(RNGManager<>::getInstance().getRandomStream("primary_particle"));
      runOneShower(obsSequence, i, E);
    }
  }

  output.endOfLibrary();
  return EXIT_SUCCESS;
}
