package core.levelgen;

import static core.Types.RESOURCE.*;
import static core.Types.TERRAIN.*;
import static core.Types.TRIBE.*;
//import static core.Types.RESOURCE.*;
import core.TribesConfig;
import core.Types;
import org.json.JSONObject;
import utils.file.IO;

import java.io.FileWriter;
import java.util.*;

/**
 * This is a Java port of the level generator created for the game Polytopia adapted to our format.
 * Original source: https://github.com/QuasiStellar/Polytopia-Map-Generator.
 */

//TODO: Clean up and write some comments.

public class LevelGenerator {

    //Level parameters, can be changed using init().
    private int mapSize;
    private int smoothing;
    private int relief;
    private double initialLand;
    private double landCoefficient;
    private String[] level;
    private Types.TRIBE[] tribes;
    private double BORDER_EXPANSION = 1/3.0;
    private long seed;
    private Random rnd;
    private TribesConfig.MAP_TYPE mapType;
    private ArrayList<Integer> prePlacedVillageCenters = new ArrayList<>();

    //JSON that contains all the probability values for all the tribes.
    private JSONObject data;

    private boolean LEVELGEN_VERBOSE = false;

    /**
     * Constructor of the generator
     */
    public LevelGenerator(long seed) {

        this.seed = seed;
        this.rnd = new Random(seed);

        //Initialize with default values.
        init(11, 3, 4, 0.5, new Types.TRIBE[]{XIN_XI, OUMAJI}, TribesConfig.DEFAULT_MAP_TYPE);

        //Read the JSON that contains all the probability values for all the tribes.
        try {
            this.data =  new IO().readJSON("terrainProbs.json");
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Initializes the map generation parameters.
     */
    public void init(int mapSize, int smoothing, int relief, double initialLand, Types.TRIBE[] tribes) {
        init(mapSize, smoothing, relief, initialLand, tribes, TribesConfig.DEFAULT_MAP_TYPE);
    }

    public void init(int mapSize, int smoothing, int relief, double initialLand, Types.TRIBE[] tribes, TribesConfig.MAP_TYPE mapType) {
        this.mapSize = mapSize;
        this.smoothing = smoothing;
        this.relief = relief;
        this.initialLand = initialLand;
        this.level = new String[mapSize*mapSize];
        this.tribes = tribes;
        this.mapType = mapType == null ? TribesConfig.DEFAULT_MAP_TYPE : mapType;
        this.landCoefficient = (0.5 + relief) / 9;

        //Initialize the level with deep water.
        for(int i = 0; i < mapSize*mapSize; i++){ level[i] = "d: "; };
    }

    /**
     * Generates the level.
     */
    public void generate() {

        if (LEVELGEN_VERBOSE) System.out.println("Generating level with seed: " + this.seed);
        prePlacedVillageCenters.clear();
        generateBaseLand();

        // Capital distribution
        if (LEVELGEN_VERBOSE) System.out.println("Capital distribution");
        ArrayList<Integer> capitalCells = getCapitalCells();
        // Keep capital write order aligned with tribe order so row-major level loading
        // recreates players in the same order as the provided tribes array.
        Collections.sort(capitalCells);
        for (int i = 0; i < capitalCells.size(); i++) {
            writeTile((capitalCells.get(i) / mapSize) * mapSize + (capitalCells.get(i) % mapSize), ""+CITY.getMapChar(), String.valueOf(tribes[i].getKey()));
        }

        // Terrain distribution
        if (LEVELGEN_VERBOSE) System.out.println("Terrain distribution");
        ArrayList<Integer> doneTiles = new ArrayList<>();
        ArrayList<ArrayList<Integer>> activeTiles = new ArrayList<>(); // done tiles that generate terrain around them
        Types.TRIBE[] tileOwner = new Types.TRIBE[mapSize*mapSize];

        int i;
        for (i = 0; i < capitalCells.size(); i++) {
            doneTiles.add(i, capitalCells.get(i));
            ArrayList<Integer> cap = new ArrayList<>();
            cap.add(capitalCells.get(i));
            activeTiles.add(i, cap);
            tileOwner[capitalCells.get(i)] = tribes[i];
        }
        // We will start from capital tiles and evenly expand until the whole map is covered
        while (doneTiles.size() != mapSize*mapSize) {
            for (i = 0; i < tribes.length; i++) {
                if (activeTiles.get(i).size() != 0) {
                    int randNumber = randomInt(0, activeTiles.get(i).size());
                    int randCell = activeTiles.get(i).get(randNumber);

                    ArrayList<Integer> neighbours = circle(randCell, 1);

                    ArrayList<Integer> validNeighbours = new ArrayList<>();
                    for(int n : neighbours){
                        if(!doneTiles.contains(n) && getTerrain(n) != DEEP_WATER.getMapChar()){
                            validNeighbours.add(n);
                        }
                    }
                    // If there are no land tiles around, accept water tiles
                    if (validNeighbours.size() == 0) {
                        for(int n : neighbours){
                            if(!doneTiles.contains(n)){
                                validNeighbours.add(n);
                            }
                        }
                    }
                    if (validNeighbours.size() != 0) {
                        int new_rand_number = randomInt(0, validNeighbours.size());
                        int new_rand_cell = validNeighbours.get(new_rand_number);
                        tileOwner[new_rand_cell] = tribes[i];
                        activeTiles.get(i).add(new_rand_cell);
                        doneTiles.add(new_rand_cell);
                    } else {
                        activeTiles.get(i).remove(randNumber); // deactivate tiles surrounded with done tiles
                    }
                }
            }
        }

        // Generate forest/mountains using tribe-normalized quotas.
        if (LEVELGEN_VERBOSE) System.out.println("Generate forest, mountains");
        applyTerrainQuotas(tileOwner);

        ArrayList<Integer> villageMap = new ArrayList<Integer>(mapSize*mapSize);
        ArrayList<Integer> villageCenters = new ArrayList<>();

        // Initialize with zeros.
        for(i = 0; i < mapSize*mapSize; i++) {
            villageMap.add(0);
        }

        // -1 - water far away
        // 0 - far away
        // 1 - border expansion
        // 2 - initial territory
        // 3 - village
        for (int cell = 0; cell < mapSize*mapSize; cell++) {
            int row = cell / mapSize;
            int column = cell % mapSize;
            if (getTerrain(cell) == DEEP_WATER.getMapChar() || getTerrain(cell) == MOUNTAIN.getMapChar()) {
                villageMap.set(cell, -1);
            } else if (row == 0 || row == mapSize - 1 || column == 0 || column == mapSize - 1) {
                villageMap.set(cell, -1); // villages don't spawn next to the map border
            } else {
                villageMap.set(cell, 0);
            }
        }

        // Replace some ocean with shallow water
        if (LEVELGEN_VERBOSE) System.out.println("Replace some ocean with shallow water");
        for (int cell = 0; cell < mapSize*mapSize; cell++) {
            if (getTerrain(cell) == DEEP_WATER.getMapChar()) {
                for (int neighbour : crossNeighbors(cell)) {
                    char terrainN = getTerrain(neighbour);
                    if(terrainN == PLAIN.getMapChar() || terrainN == FOREST.getMapChar() || terrainN == MOUNTAIN.getMapChar()){
                        writeTile(neighbour, ""+SHALLOW_WATER.getMapChar(), null);
                        break;
                    }
                }
            }
        }

        // Mark tiles next to capitals according to the notation
        for (int capital : capitalCells){
            markVillageArea(villageMap, capital);
            villageCenters.add(capital);
        }

        // For Continents/Pangea, seed village locations are generated before capital conversion.
        // Re-introduce the non-capital seeds here so later resource passes see proper city/village bands.
        if (!prePlacedVillageCenters.isEmpty()) {
            for (int seedVillage : prePlacedVillageCenters) {
                if (villageCenters.contains(seedVillage)) continue;
                markVillageArea(villageMap, seedVillage);
                villageCenters.add(seedVillage);
            }
        }

        // Suburbs are modeled as special villages near capitals for Lakes/Archipelago.
        if (usesSuburbPhase()) {
            for (int capital : capitalCells) {
                int suburbsTarget = randomInt(1, 3); // 1-2 suburbs per capital.
                int placed = 0;
                ArrayList<Integer> suburbanCandidates = getVillageCandidates(
                        1, villageCenters, 2, new HashSet<>(circle(capital, 2))
                );
                while (placed < suburbsTarget && !suburbanCandidates.isEmpty()) {
                    int idx = randomInt(0, suburbanCandidates.size());
                    int village = suburbanCandidates.remove(idx);
                    if (!isVillageSpacingValid(village, villageCenters, 2)) continue;
                    markVillageArea(villageMap, village);
                    villageCenters.add(village);
                    placed++;
                }
            }
        }

        // Pre-terrain villages (Lakes/Archipelago/Waterworld style) are approximated here
        // as an additional early village pass before final saturation.
        if (usesPreTerrainVillagePhase()) {
            int base = (int) Math.pow(Math.floor(mapSize / 3.0), 2);
            int preTarget = Math.max(0, (int) Math.round((base - villageCenters.size()) * getPreTerrainVillageDensity()));
            ArrayList<Integer> preCandidates = getVillageCandidates(1, villageCenters, 2, null);
            while (preTarget > 0 && !preCandidates.isEmpty()) {
                int idx = randomInt(0, preCandidates.size());
                int village = preCandidates.remove(idx);
                if (!isVillageSpacingValid(village, villageCenters, 2)) continue;
                markVillageArea(villageMap, village);
                villageCenters.add(village);
                preTarget--;
            }
        }

        // Generate villages & mark tiles next to them
        // We will place villages until there are none of "far away" (villageMap == 0) tiles.
        while (true) {
            ArrayList<Integer> postCandidates = getVillageCandidates(2, villageCenters, 2, null);
            if (postCandidates.isEmpty()) break;
            int new_village = postCandidates.get(randomInt(0, postCandidates.size()));
            markVillageArea(villageMap, new_village);
            villageCenters.add(new_village);
        }

        // Continents and Pangea include fixed tiny-island villages by map size.
        if (mapType == TribesConfig.MAP_TYPE.CONTINENTS || mapType == TribesConfig.MAP_TYPE.PANGEA) {
            addTinyIslandVillages(villageMap, villageCenters, getTinyIslandVillageTarget());
        }

        // Stamp villages into terrain before resource assignment.
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (villageMap.get(cell) == 3 && getTerrain(cell) != CITY.getMapChar()) {
                writeTile(cell, "" + VILLAGE.getMapChar(), null);
            }
        }

        // Generate resources using per-tribe, per-band quotas.
        if (LEVELGEN_VERBOSE) System.out.println("Generate resources");
        applyResourceQuotas(villageMap, tileOwner);

        // Ruins generation.
        if (LEVELGEN_VERBOSE) System.out.println("Ruins generation");

        int ruins_number = (int) Math.round((mapSize*mapSize)/40.0);
        int water_ruins_number = (int) Math.round(ruins_number/3.0);
        int ruins_count = 0;
        int water_ruins_count = 0;


        while (ruins_count < ruins_number) {

            // We are reusing villageMap even though it is irrelevant in this context but it has useful info for ruin placement.
            ArrayList<Integer> ruinCandidates = new ArrayList<>();
            for(i=0; i < villageMap.size(); i++) {
                int cell = villageMap.get(i);
                if(cell == 0 || cell == 1 || cell == -1) {
                    ruinCandidates.add(i);
                }
            }

            int ruin = ruinCandidates.get(randomInt(0,ruinCandidates.size()));
            if (getTerrain(ruin) != SHALLOW_WATER.getMapChar() && (water_ruins_count < water_ruins_number || getTerrain(ruin) != DEEP_WATER.getMapChar())) {
                writeTile(ruin, null, ""+RUINS.getMapChar());
                if (getTerrain(ruin) == DEEP_WATER.getMapChar()) {
                    water_ruins_count++;
                }

                //This avoids having contiguous ruins and favours dispersion.
                for (int neighbour : circle(villageMap.get(ruin), 1)) {
                    villageMap.set(neighbour, Math.max(villageMap.get(neighbour), 2));
                }

                ruins_count++;
            }
        }

        // Re-adjust starting tiles around capitals
        if (LEVELGEN_VERBOSE) System.out.println("Re-adjust starting tiles around capitals");
        for(int capital : capitalCells) {
            int owner = Integer.parseInt(getResource(capital));

            if(owner == IMPERIUS.getKey()) {
                postGenerate(FRUIT.getMapChar(), PLAIN.getMapChar(), 2, capital);
            } else if(owner == BARDUR.getKey()) {
                postGenerate(ANIMAL.getMapChar(), FOREST.getMapChar(), 2, capital);
            }
        }
    }

    private void generateBaseLand() {
        if (mapType == TribesConfig.MAP_TYPE.PANGEA) {
            generatePangeaLand();
        } else if (mapType == TribesConfig.MAP_TYPE.CONTINENTS) {
            generateContinentsLand();
        } else {
            generateGenericSmoothedLand();
        }
    }

    private void generateGenericSmoothedLand() {
        if (LEVELGEN_VERBOSE) System.out.println("Randomly replace half of the tiles with ground.");
        int i = 0;
        while (i < mapSize * mapSize * initialLand) {
            int index = randomInt(0, mapSize * mapSize);
            if (getTerrain(index) == DEEP_WATER.getMapChar()) {
                i++;
                writeTile(index, "" + PLAIN.getMapChar(), null);
            }
        }

        if (LEVELGEN_VERBOSE) System.out.println("Turning random water/ground grid into something smooth.");
        ArrayList<Integer> toBeGround = new ArrayList<>();
        for (i = 0; i < smoothing; i++) {
            toBeGround.clear();
            for (int cell = 0; cell < mapSize * mapSize; cell++) {
                int water_count = 0;
                int tile_count = 0;
                for (int n : disk(cell, 1)) {
                    if (getTerrain(n) == DEEP_WATER.getMapChar()) {
                        water_count++;
                    }
                    tile_count++;
                }
                if (water_count / (double) tile_count <= landCoefficient) {
                    toBeGround.add(cell);
                }
            }

            for (int cell = 0; cell < mapSize * mapSize; cell++) {
                if (toBeGround.contains(cell)) {
                    writeTile(cell, "" + PLAIN.getMapChar(), null);
                } else {
                    writeTile(cell, "" + DEEP_WATER.getMapChar(), null);
                }
            }
        }
    }

    private void generatePangeaLand() {
        int total = mapSize * mapSize;
        int targetLand = Math.max(tribes.length * 10, (int) Math.round(total * initialLand));
        int center = (mapSize / 2) * mapSize + (mapSize / 2);
        growLandCluster(center, targetLand, true);
    }

    private void generateContinentsLand() {
        int total = mapSize * mapSize;
        int targetLand = Math.max(tribes.length * 10, (int) Math.round(total * initialLand));
        int continentCount = Math.max(2, Math.min(6, tribes.length + (mapSize >= 20 ? 1 : 0)));
        int remaining = targetLand;

        ArrayList<Integer> centers = new ArrayList<>();
        int attempts = 0;
        while (centers.size() < continentCount && attempts < continentCount * 20) {
            attempts++;
            int row = randomInt(2, mapSize - 2);
            int col = randomInt(2, mapSize - 2);
            int idx = row * mapSize + col;
            boolean farEnough = true;
            for (int c : centers) {
                if (distance(c, idx, mapSize) < Math.max(3, mapSize / 6)) {
                    farEnough = false;
                    break;
                }
            }
            if (farEnough) centers.add(idx);
        }
        if (centers.isEmpty()) centers.add((mapSize / 2) * mapSize + (mapSize / 2));

        for (int i = 0; i < centers.size(); i++) {
            int chunksLeft = centers.size() - i;
            int size = i == centers.size() - 1 ? remaining : Math.max(8, remaining / chunksLeft);
            growLandCluster(centers.get(i), size, false);
            remaining -= size;
        }
    }

    private void growLandCluster(int start, int landCount, boolean radialBias) {
        int placed = 0;
        HashSet<Integer> cluster = new HashSet<>();
        ArrayList<Integer> frontier = new ArrayList<>();
        frontier.add(start);
        cluster.add(start);
        writeTile(start, "" + PLAIN.getMapChar(), null);
        placed++;

        while (placed < landCount && !frontier.isEmpty()) {
            int fIdx = randomInt(0, frontier.size());
            int cell = frontier.get(fIdx);
            ArrayList<Integer> neighbours = crossNeighbors(cell);
            Collections.shuffle(neighbours, rnd);
            boolean expanded = false;
            for (int n : neighbours) {
                if (cluster.contains(n)) continue;
                int row = n / mapSize;
                int col = n % mapSize;
                if (row <= 0 || row >= mapSize - 1 || col <= 0 || col >= mapSize - 1) continue;
                if (radialBias) {
                    int cr = mapSize / 2;
                    int cc = mapSize / 2;
                    int dr = row - cr;
                    int dc = col - cc;
                    double radial = Math.sqrt(dr * dr + dc * dc);
                    double maxRadial = Math.sqrt(2) * (mapSize / 2.0);
                    double p = 1.0 - (radial / Math.max(1.0, maxRadial));
                    if (rnd.nextDouble() > clamp01(0.25 + 0.75 * p)) continue;
                }
                cluster.add(n);
                frontier.add(n);
                writeTile(n, "" + PLAIN.getMapChar(), null);
                placed++;
                expanded = true;
                if (placed >= landCount) break;
            }
            if (!expanded) {
                frontier.remove(fIdx);
            }
        }
    }

    private boolean usesQuadrantCapitals() {
        return mapType == TribesConfig.MAP_TYPE.DRYLANDS
                || mapType == TribesConfig.MAP_TYPE.LAKES
                || mapType == TribesConfig.MAP_TYPE.ARCHIPELAGO
                || mapType == TribesConfig.MAP_TYPE.WATERWORLD;
    }

    private boolean usesSuburbPhase() {
        return mapType == TribesConfig.MAP_TYPE.LAKES
                || mapType == TribesConfig.MAP_TYPE.ARCHIPELAGO;
    }

    private boolean usesPreTerrainVillagePhase() {
        return mapType == TribesConfig.MAP_TYPE.LAKES
                || mapType == TribesConfig.MAP_TYPE.ARCHIPELAGO
                || mapType == TribesConfig.MAP_TYPE.WATERWORLD;
    }

    private double getPreTerrainVillageDensity() {
        if (mapType == TribesConfig.MAP_TYPE.WATERWORLD) return 0.1;
        if (mapType == TribesConfig.MAP_TYPE.LAKES || mapType == TribesConfig.MAP_TYPE.ARCHIPELAGO) return 0.3;
        return 0.0;
    }

    private ArrayList<Integer> getCapitalCells() {
        if (mapType == TribesConfig.MAP_TYPE.CONTINENTS || mapType == TribesConfig.MAP_TYPE.PANGEA) {
            return getCapitalCellsFromVillageConversion();
        }
        if (usesQuadrantCapitals()) {
            return getCapitalCellsByQuadrants();
        }
        return getCapitalCellsByDistance();
    }

    private ArrayList<Integer> getCapitalCellsFromVillageConversion() {
        HashSet<Integer> villageSeeds = new HashSet<>();
        ArrayList<ArrayList<Integer>> components = getPlainLandComponents();

        // Ensure at least one seed village per sizeable land component.
        for (ArrayList<Integer> component : components) {
            if (component.size() < 6) continue;
            ArrayList<Integer> local = new ArrayList<>();
            for (int cell : component) {
                int row = cell / mapSize;
                int col = cell % mapSize;
                if (row <= 0 || row >= mapSize - 1 || col <= 0 || col >= mapSize - 1) continue;
                local.add(cell);
            }
            if (!local.isEmpty()) {
                villageSeeds.add(local.get(randomInt(0, local.size())));
            }
        }

        // Fill additional villages globally by spacing until no candidates remain.
        ArrayList<Integer> villages = new ArrayList<>(villageSeeds);
        ArrayList<Integer> candidates = getVillageCandidates(1, villages, 2, null);
        while (!candidates.isEmpty()) {
            int cell = candidates.get(randomInt(0, candidates.size()));
            villages.add(cell);
            villageSeeds.add(cell);
            candidates = getVillageCandidates(1, villages, 2, null);
        }

        prePlacedVillageCenters = new ArrayList<>(villages);

        ArrayList<Integer> capitals = new ArrayList<>();
        ArrayList<Integer> candidatesForCapitals = new ArrayList<>(villages);
        for (int t = 0; t < tribes.length && !candidatesForCapitals.isEmpty(); t++) {
            int bestCell = -1;
            double bestScore = -Double.MAX_VALUE;
            for (int cell : candidatesForCapitals) {
                double minDist = capitals.isEmpty() ? mapSize : Double.MAX_VALUE;
                for (int c : capitals) {
                    minDist = Math.min(minDist, distance(cell, c, mapSize));
                }
                double score = minDist;
                if (mapType == TribesConfig.MAP_TYPE.PANGEA && isCoastal(cell)) {
                    score += 1.5; // prefer coastal starts on Pangea
                }
                score += rnd.nextDouble() * 0.05; // tie-breaker noise
                if (score > bestScore) {
                    bestScore = score;
                    bestCell = cell;
                }
            }
            if (bestCell >= 0) {
                capitals.add(bestCell);
                candidatesForCapitals.remove((Integer) bestCell);
            }
        }

        if (capitals.size() < tribes.length) {
            ArrayList<Integer> fallback = getCapitalCellsByDistance();
            for (int cell : fallback) {
                if (capitals.size() >= tribes.length) break;
                if (!capitals.contains(cell)) capitals.add(cell);
            }
        }

        return capitals;
    }

    private ArrayList<Integer> getCapitalCellsByDistance() {
        ArrayList<Integer> capitalCells = new ArrayList<>();
        HashMap<Integer, Integer> capitalMap = new HashMap<>();
        for (Types.TRIBE tribe : tribes) {
            for (int row = 2; row < mapSize - 2; row++) {
                for (int column = 2; column < mapSize - 2; column++) {
                    if (getTerrain(row * mapSize + column) == PLAIN.getMapChar()) {
                        capitalMap.put(row * mapSize + column, 0);
                    }
                }
            }
        }

        for (Types.TRIBE tribe : tribes) {
            int max = 0;
            Iterator capitalIterator = capitalMap.entrySet().iterator();
            while (capitalIterator.hasNext()) {
                Map.Entry cell = (Map.Entry) capitalIterator.next();
                cell.setValue(mapSize);
                for (int capitalCell : capitalCells) {
                    cell.setValue(Math.min((int) cell.getValue(), distance((int) cell.getKey(), capitalCell, mapSize)));
                }
                max = Math.max(max, (int) cell.getValue());
            }

            int len = 0;
            capitalIterator = capitalMap.entrySet().iterator();
            while (capitalIterator.hasNext()) {
                Map.Entry cell = (Map.Entry) capitalIterator.next();
                if ((int) cell.getValue() == max) {
                    len++;
                }
            }
            if (len == 0) {
                break;
            }

            int randCell = randomInt(0, len);
            capitalIterator = capitalMap.entrySet().iterator();
            while (capitalIterator.hasNext()) {
                Map.Entry cell = (Map.Entry) capitalIterator.next();
                if ((int) cell.getValue() == max) {
                    if (randCell == 0) {
                        capitalCells.add((int) cell.getKey());
                        if (LEVELGEN_VERBOSE) {
                            System.out.println("Adding a capital for tribe " + tribe + " at tile " + (int) cell.getKey()
                                    + " with a max distance of " + cell.getValue());
                        }
                        break;
                    }
                    randCell--;
                }
            }
        }

        if (capitalCells.size() < tribes.length) {
            ArrayList<Integer> fallback = getCapitalCellsByQuadrants();
            for (int cell : fallback) {
                if (capitalCells.size() >= tribes.length) break;
                if (!capitalCells.contains(cell)) capitalCells.add(cell);
            }
        }

        return capitalCells;
    }

    private ArrayList<Integer> getCapitalCellsByQuadrants() {
        ArrayList<Integer> candidates = new ArrayList<>();
        for (int row = 1; row < mapSize - 1; row++) {
            for (int column = 1; column < mapSize - 1; column++) {
                int idx = row * mapSize + column;
                if (getTerrain(idx) == PLAIN.getMapChar()) {
                    candidates.add(idx);
                }
            }
        }

        int domainsPerAxis = tribes.length <= 4 ? 2 : (tribes.length <= 9 ? 3 : 4);
        int nDomains = domainsPerAxis * domainsPerAxis;
        ArrayList<Integer> domainIds = new ArrayList<>();
        for (int i = 0; i < nDomains; i++) domainIds.add(i);
        Collections.shuffle(domainIds, rnd);

        ArrayList<Integer> capitals = new ArrayList<>();
        for (int tribeIdx = 0; tribeIdx < tribes.length; tribeIdx++) {
            int domainId = domainIds.get(tribeIdx % domainIds.size());
            int domainRow = domainId / domainsPerAxis;
            int domainCol = domainId % domainsPerAxis;

            int rowStart = (domainRow * mapSize) / domainsPerAxis;
            int rowEnd = ((domainRow + 1) * mapSize) / domainsPerAxis - 1;
            int colStart = (domainCol * mapSize) / domainsPerAxis;
            int colEnd = ((domainCol + 1) * mapSize) / domainsPerAxis - 1;

            ArrayList<Integer> localCandidates = new ArrayList<>();
            for (int row = Math.max(1, rowStart); row <= Math.min(mapSize - 2, rowEnd); row++) {
                for (int col = Math.max(1, colStart); col <= Math.min(mapSize - 2, colEnd); col++) {
                    int idx = row * mapSize + col;
                    if (getTerrain(idx) != PLAIN.getMapChar()) continue;
                    if (capitals.contains(idx)) continue;
                    localCandidates.add(idx);
                }
            }

            int picked = -1;
            if (!localCandidates.isEmpty()) {
                picked = localCandidates.get(randomInt(0, localCandidates.size()));
            } else {
                ArrayList<Integer> fallback = new ArrayList<>();
                for (int idx : candidates) {
                    if (!capitals.contains(idx)) fallback.add(idx);
                }
                if (!fallback.isEmpty()) {
                    picked = fallback.get(randomInt(0, fallback.size()));
                }
            }

            if (picked >= 0) capitals.add(picked);
        }
        return capitals;
    }

    private void markVillageArea(ArrayList<Integer> villageMap, int center) {
        villageMap.set(center, 3);
        for (int cell : circle(center, 1)) {
            villageMap.set(cell, Math.max(villageMap.get(cell), 2));
        }
        for (int cell : circle(center, 2)) {
            villageMap.set(cell, Math.max(villageMap.get(cell), 1));
        }
    }

    private boolean isVillageSpacingValid(int cell, ArrayList<Integer> villageCenters, int minDistance) {
        for (int placed : villageCenters) {
            if (distance(cell, placed, mapSize) <= minDistance) {
                return false;
            }
        }
        return true;
    }

    private ArrayList<Integer> getVillageCandidates(int minEdgeDistance, ArrayList<Integer> villageCenters,
                                                    int minVillageDistance, Set<Integer> preferredCells) {
        ArrayList<Integer> candidates = new ArrayList<>();
        Set<Integer> preference = preferredCells == null ? Collections.emptySet() : preferredCells;
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            int row = cell / mapSize;
            int col = cell % mapSize;
            if (row < minEdgeDistance || row > mapSize - 1 - minEdgeDistance
                    || col < minEdgeDistance || col > mapSize - 1 - minEdgeDistance) {
                continue;
            }
            char terrain = getTerrain(cell);
            if (terrain == DEEP_WATER.getMapChar() || terrain == MOUNTAIN.getMapChar()) continue;
            if (!preference.isEmpty() && !preference.contains(cell)) continue;
            if (!isVillageSpacingValid(cell, villageCenters, minVillageDistance)) continue;
            candidates.add(cell);
        }
        return candidates;
    }

    private boolean isCoastal(int cell) {
        for (int n : crossNeighbors(cell)) {
            char t = getTerrain(n);
            if (t == DEEP_WATER.getMapChar() || t == SHALLOW_WATER.getMapChar()) return true;
        }
        return false;
    }

    private ArrayList<ArrayList<Integer>> getPlainLandComponents() {
        ArrayList<ArrayList<Integer>> components = new ArrayList<>();
        boolean[] visited = new boolean[mapSize * mapSize];
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (visited[cell]) continue;
            if (getTerrain(cell) != PLAIN.getMapChar()) continue;
            ArrayList<Integer> component = new ArrayList<>();
            ArrayDeque<Integer> q = new ArrayDeque<>();
            q.add(cell);
            visited[cell] = true;
            while (!q.isEmpty()) {
                int cur = q.poll();
                component.add(cur);
                for (int n : crossNeighbors(cur)) {
                    if (visited[n]) continue;
                    if (getTerrain(n) != PLAIN.getMapChar()) continue;
                    visited[n] = true;
                    q.add(n);
                }
            }
            if (!component.isEmpty()) components.add(component);
        }
        return components;
    }

    private int getTinyIslandVillageTarget() {
        if (mapSize <= 11) return 0;
        if (mapSize <= 14) return 1;
        if (mapSize <= 16) return 2;
        if (mapSize <= 18) return 3;
        if (mapSize <= 20) return 4;
        return Math.max(5, (int) Math.round(mapSize * mapSize / 100.0));
    }

    private void addTinyIslandVillages(ArrayList<Integer> villageMap, ArrayList<Integer> villageCenters, int target) {
        int placed = 0;
        int attempts = 0;
        while (placed < target && attempts < mapSize * mapSize * 8) {
            attempts++;
            int row = randomInt(2, mapSize - 2);
            int col = randomInt(2, mapSize - 2);
            int cell = row * mapSize + col;
            if (villageCenters.contains(cell)) continue;
            if (!isVillageSpacingValid(cell, villageCenters, 2)) continue;
            if (getTerrain(cell) != DEEP_WATER.getMapChar()) continue;

            boolean isolated = true;
            for (int n : disk(cell, 2)) {
                if (n == cell) continue;
                char t = getTerrain(n);
                if (t == PLAIN.getMapChar() || t == FOREST.getMapChar() || t == MOUNTAIN.getMapChar()
                        || t == CITY.getMapChar() || t == VILLAGE.getMapChar()) {
                    isolated = false;
                    break;
                }
            }
            if (!isolated) continue;

            writeTile(cell, "" + PLAIN.getMapChar(), null);
            markVillageArea(villageMap, cell);
            villageCenters.add(cell);
            placed++;
        }
    }

    private void applyTerrainQuotas(Types.TRIBE[] tileOwner) {
        for (Types.TRIBE tribe : tribes) {
            ArrayList<Integer> candidates = new ArrayList<>();
            for (int cell = 0; cell < mapSize * mapSize; cell++) {
                if (tileOwner[cell] != tribe) continue;
                if (getTerrain(cell) != PLAIN.getMapChar()) continue;
                candidates.add(cell);
            }
            if (candidates.isEmpty()) continue;

            double baseMountain = 0.14;
            double baseForest = 0.38;
            double mountainRate = clamp01(baseMountain * getTribeProb("MOUNTAIN", tribe));
            if (mountainRate > 0.95) mountainRate = 0.95;
            double forestScaled = baseForest * getTribeProb("FOREST", tribe);
            double forestRate = forestScaled * ((1.0 - mountainRate) / (1.0 - baseMountain));
            forestRate = clamp01(forestRate);
            if (forestRate > (1.0 - mountainRate)) forestRate = 1.0 - mountainRate;

            int[] counts = allocateCountsByWeights(candidates.size(), new double[]{
                    mountainRate, forestRate, Math.max(0.0, 1.0 - mountainRate - forestRate)
            });
            Collections.shuffle(candidates, rnd);
            int idx = 0;
            for (int n = 0; n < counts[0] && idx < candidates.size(); n++, idx++) {
                writeTile(candidates.get(idx), "" + MOUNTAIN.getMapChar(), null);
            }
            for (int n = 0; n < counts[1] && idx < candidates.size(); n++, idx++) {
                writeTile(candidates.get(idx), "" + FOREST.getMapChar(), null);
            }
        }
    }

    private void applyResourceQuotas(ArrayList<Integer> villageMap, Types.TRIBE[] tileOwner) {
        for (Types.TRIBE tribe : tribes) {
            applyResourceQuotasForBand(villageMap, tileOwner, tribe, 2);
            applyResourceQuotasForBand(villageMap, tileOwner, tribe, 1);
        }
    }

    private void applyResourceQuotasForBand(ArrayList<Integer> villageMap, Types.TRIBE[] tileOwner,
                                            Types.TRIBE tribe, int bandState) {
        boolean inner = bandState == 2;

        ArrayList<Integer> plainCells = getOwnedCellsByTerrainBand(villageMap, tileOwner, tribe, PLAIN.getMapChar(), bandState);
        if (!plainCells.isEmpty()) {
            double fruitWeight = (inner ? 18.0 : 6.0) * getTribeProb("FRUIT", tribe);
            double cropWeight = (inner ? 18.0 : 6.0) * getTribeProb("CROPS", tribe);
            double emptyWeight = inner ? 12.0 : 36.0;
            int[] counts = allocateCountsByWeights(plainCells.size(), new double[]{fruitWeight, cropWeight, emptyWeight});
            assignResourceCounts(plainCells, new char[]{FRUIT.getMapChar(), CROPS.getMapChar()}, new int[]{counts[0], counts[1]});
        }

        ArrayList<Integer> forestCells = getOwnedCellsByTerrainBand(villageMap, tileOwner, tribe, FOREST.getMapChar(), bandState);
        if (!forestCells.isEmpty()) {
            double animalWeight = (inner ? 19.0 : 6.0) * getTribeProb("ANIMAL", tribe);
            double emptyWeight = inner ? 19.0 : 32.0;
            int[] counts = allocateCountsByWeights(forestCells.size(), new double[]{animalWeight, emptyWeight});
            assignResourceCounts(forestCells, new char[]{ANIMAL.getMapChar()}, new int[]{counts[0]});
        }

        ArrayList<Integer> mountainCells = getOwnedCellsByTerrainBand(villageMap, tileOwner, tribe, MOUNTAIN.getMapChar(), bandState);
        if (!mountainCells.isEmpty()) {
            double oreWeight = (inner ? 11.0 : 3.0) * getTribeProb("ORE", tribe);
            double emptyWeight = inner ? 3.0 : 11.0;
            int[] counts = allocateCountsByWeights(mountainCells.size(), new double[]{oreWeight, emptyWeight});
            assignResourceCounts(mountainCells, new char[]{ORE.getMapChar()}, new int[]{counts[0]});
        }

        ArrayList<Integer> shallowCells = getOwnedCellsByTerrainBand(villageMap, tileOwner, tribe, SHALLOW_WATER.getMapChar(), bandState);
        if (!shallowCells.isEmpty()) {
            double fishRate = clamp01(0.5 * getTribeProb("FISH", tribe));
            int[] counts = allocateCountsByWeights(shallowCells.size(), new double[]{fishRate, 1.0 - fishRate});
            assignResourceCounts(shallowCells, new char[]{FISH.getMapChar()}, new int[]{counts[0]});
        }

        ArrayList<Integer> deepCells = getOwnedCellsByTerrainBand(villageMap, tileOwner, tribe, DEEP_WATER.getMapChar(), bandState);
        if (!deepCells.isEmpty()) {
            double whalesRate = clamp01(getBaseProb("WHALES") * getTribeProb("WHALES", tribe));
            int[] counts = allocateCountsByWeights(deepCells.size(), new double[]{whalesRate, 1.0 - whalesRate});
            assignResourceCounts(deepCells, new char[]{WHALES.getMapChar()}, new int[]{counts[0]});
        }
    }

    private ArrayList<Integer> getOwnedCellsByTerrainBand(ArrayList<Integer> villageMap, Types.TRIBE[] tileOwner,
                                                          Types.TRIBE tribe, char terrain, int bandState) {
        ArrayList<Integer> cells = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (tileOwner[cell] != tribe) continue;
            if (villageMap.get(cell) != bandState) continue;
            if (getTerrain(cell) != terrain) continue;
            cells.add(cell);
        }
        return cells;
    }

    private void assignResourceCounts(ArrayList<Integer> cells, char[] resources, int[] counts) {
        if (cells.isEmpty() || resources.length == 0 || counts.length == 0) return;
        ArrayList<Integer> shuffled = new ArrayList<>(cells);
        Collections.shuffle(shuffled, rnd);
        int idx = 0;
        for (int i = 0; i < resources.length && i < counts.length; i++) {
            for (int n = 0; n < counts[i] && idx < shuffled.size(); n++, idx++) {
                writeTile(shuffled.get(idx), null, "" + resources[i]);
            }
        }
    }

    private int[] allocateCountsByWeights(int total, double[] weights) {
        int[] counts = new int[weights.length];
        if (total <= 0 || weights.length == 0) return counts;

        double sum = 0.0;
        double[] positive = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            positive[i] = Math.max(0.0, weights[i]);
            sum += positive[i];
        }
        if (sum <= 0.0) {
            counts[weights.length - 1] = total;
            return counts;
        }

        double[] frac = new double[weights.length];
        int assigned = 0;
        for (int i = 0; i < weights.length; i++) {
            double exact = total * (positive[i] / sum);
            counts[i] = (int) Math.floor(exact);
            frac[i] = exact - counts[i];
            assigned += counts[i];
        }
        int remaining = total - assigned;
        while (remaining > 0) {
            int best = 0;
            for (int i = 1; i < frac.length; i++) {
                if (frac[i] > frac[best]) best = i;
            }
            counts[best]++;
            frac[best] = 0.0;
            remaining--;
        }
        return counts;
    }

    private double clamp01(double v) {
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    /**
     * Counts the instances of a resource that exists on the starting tiles that surround a capital.
     * @param resource the resource to be counted.
     * @param capital the index of the capital.
     * @return the resource counter.
     */
    public int checkResources(char resource, int capital) {
        int resources = 0;
        for (int neighbour : circle(capital, 1)) {
            String resourceStr = getResource(neighbour);
            if(resourceStr.length() > 0 && resourceStr.charAt(0) == resource){
                resources++;
            }
        }
        return resources;
    }

    /**
     * Adds the required amount of a specific resource on specific type of terrain
     * in the starting tiles that surround a capital.
     * @param resource the resource to be added.
     * @param terrain the terrain on top of which the resource will be added.
     * @param quantity the amount to be tiles that must have this terrain + resource combination.
     */
    public void postGenerate(char resource, char terrain, int quantity, int capital) {
        int resources = checkResources(resource, capital);
        while (resources < quantity) {
            int pos = randomInt(0, 8);
            ArrayList<Integer> territory = circle(capital, 1);
            writeTile(territory.get(pos), ""+terrain, ""+resource);
            for (int neighbour : crossNeighbors(territory.get(pos))) {
                if (getTerrain(neighbour) == DEEP_WATER.getMapChar()) {
                    writeTile(neighbour, ""+SHALLOW_WATER.getMapChar(), null);
                }
            }
            resources = checkResources(resource, capital);
        }
    }
    /**
     * Utility function used in the generator.
     */
    public boolean proc(ArrayList<Integer> villageMap, int cell, double probability) {
        return (villageMap.get(cell) == 2 && rnd.nextDouble() < probability) || (villageMap.get(cell) == 1 && rnd.nextDouble() < probability * BORDER_EXPANSION);
    }

    /**
     * Reads the JSON configuration file and returns the probability of a terrain or a resource for a specific tribe.
     * @param name the name of the terrain or resource.
     * @param tribe the name of the tribe.
     * @return the probability.
     */
    public double getTribeProb(String name, Types.TRIBE tribe) {
        if(tribe == null) {
            return 1.0;
        } else {
            return data.getJSONObject(name.toString()).getDouble(tribe.toString());
        }
    }

    /**
     * Reads the JSON configuration file and returns the base probability of a specific terrain or resource.
     * @param name the name of the terrain or resource.
     * @return the base probability.
     */
    public double getBaseProb(String name) {
        return data.getJSONObject(name.toString()).getDouble("BASE");
    }

    /**
     * Writes a level tile at a specified position (consult the TERRAIN and RESOURCE enums).
     * @param index the index of the tile that needs to be written.
     * @param terrain the desired type of terrain.
     * @param resource the desired type of resource.
     */
    public void writeTile(int index, String terrain, String resource) {
        if(terrain == null) {
            level[index] = "" + getTerrain(index) + ':' + resource;
        }else if(resource == null) {
            level[index] = "" + terrain + ':' + getResource(index);
        }else {
            level[index] = "" + terrain + ':' + resource;
        }
    }

    /**
     * Returns a tile's terrain at a specified position.
     * @param index the desired position.
     * @return the character that represents the specific terrain (consult TERRAIN enum).
     */
    public char getTerrain(int index) {
        return level[index].split(":")[0].charAt(0);
    }


//    public char getResource(int index) {
//        return level[index].split(":")[1].charAt(0);
////        try {
////            return level[index].split(":")[1].charAt(0);
////        } catch(Exception e) {
////            return '';
////        }
//    }
    /**
     * Returns a tile's resource at a specified position.
     * @param index the desired position.
     * @return the character that represents the specific resource (consult RESOURCE enum).
     */
    public String getResource(int index)
    {
        String[] pieces = level[index].split(":");
        if(pieces.length > 1)
            if(pieces[1].charAt(0) == ' ')
                return "";
            else return pieces[1];
        else return "";
    }

    /**
     * Returns a random int in the range [min, max).
     * @param min lower bound (inclusive).
     * @param max upper bound (exclusive).
     * @return a random int.
     */
    public int randomInt(int min, int max) {
        return (int) Math.floor(min + rnd.nextDouble() * (max - min));
    }

    /**
     * Returns the indices of the map that lie on a circle.
     * @param center center of the circle.
     * @param radius radius of the circle.
     * @return an ArrayList of indices.
     */
    public ArrayList<Integer> circle(int center, int radius) {
        ArrayList<Integer> circle = new ArrayList<>();
        int row = center / mapSize;
        int column = center % mapSize;
        int i = row - radius;
        if (i >= 0 && i < mapSize) {
            for (int j = column - radius; j < column + radius; j++) {
                if (j >= 0 && j < mapSize) {
                    circle.add(i * mapSize + j);
                }
            }
        }
        i = row + radius;
        if (i >= 0 && i < mapSize) {
            for (int j = column + radius; j > column - radius; j--) {
                if (j >= 0 && j < mapSize) {
                    circle.add(i * mapSize + j);
                }
            }
        }
        int j = column - radius;
        if (j >= 0 && j < mapSize) {
            for (i = row + radius; i > row - radius; i--) {
                if (i >= 0 && i < mapSize) {
                    circle.add(i * mapSize + j);
                }
            }
        }
        j = column + radius;
        if (j >= 0 && j < mapSize) {
            for (i = row - radius; i < row + radius; i++) {
                if (i >= 0 && i < mapSize) {
                    circle.add(i * mapSize + j);
                }
            }
        }
        return circle;
    }

    /**
     * Returns the indices of the map that lie on and inside a circle including the center.
     * @param center center of the circle.
     * @param radius radius of the circle.
     * @return an ArrayList of indices.
     */
    public ArrayList<Integer> disk(int center, int radius) {
        ArrayList<Integer> round = new ArrayList<>();
        for (int r = 1; r <= radius; r++) {
            round.addAll(circle(center, r));
        }
        round.add(center);
        return round;
    }

    /**
     * Returns the indices of the map that lie on the cross pattern.
     * @param center center of the cross.
     * @return an ArrayList of indices.
     */
    public ArrayList<Integer> crossNeighbors(int center) {
        ArrayList<Integer> plus_sign = new ArrayList<>();
        int row = center / mapSize;
        int column = center % mapSize;
        if (column > 0) {
            plus_sign.add(center - 1);
        }
        if (column < mapSize - 1) {
            plus_sign.add(center + 1);
        }
        if (row > 0) {
            plus_sign.add(center - mapSize);
        }
        if (row < mapSize - 1) {
            plus_sign.add(center + mapSize);
        }
        return plus_sign;
    }

    // we use pythagorean distances
    public int distance(int a, int b, int size) {
        int ax = a % size;
        int ay = a / size;
        int bx = b % size;
        int by = b / size;
        return Math.max(Math.abs(ax - bx), Math.abs(ay - by));
    }

    /**
     * Saves the generated level into a .csv format readable by the Tribes framework.
     * @param filename path to save the level.
     */
    public void toCSV(String filename) {
        try {
            FileWriter writer = new FileWriter(filename);
            writer.append(level[0]);
            writer.append(',');
            for(int i = 1; i < mapSize*mapSize; i++) {
                if(i % mapSize == 0) {
                    writer.append('\n');
                    writer.append(level[i]);
                    writer.append(',');
                } else if(i % mapSize == mapSize - 1) {
                    writer.append(level[i]);
                }else {
                    writer.append(level[i]);
                    writer.append(',');
                }
            }
            writer.flush();
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Prints the generated level in console.
     */
    public void print() {
        StringBuffer writer = new StringBuffer();
        writer.append(level[0]);
        writer.append(',');
        for (int i = 1; i < mapSize * mapSize; i++) {
            if (i % mapSize == 0) {
                writer.append('\n');
                writer.append(level[i]);
                writer.append(',');
            } else if (i % mapSize == mapSize - 1) {
                writer.append(level[i]);
            } else {
                writer.append(level[i]);
                writer.append(',');
            }
        }
        System.out.println(writer.toString());
    }

    /**
     * Returns the generated level into a format readable by the Tribes framework.
     */
    public String[] gelLevelLines()
    {
        String[] allLines = new String[mapSize];
        int lineCounter = 0;

        StringBuffer line = new StringBuffer();
        line.append(level[0]);
        line.append(',');
        for (int i = 1; i < mapSize * mapSize; i++) {
            if (i % mapSize == mapSize - 1) {
                line.append(level[i]);
                allLines[lineCounter] = line.toString();
                lineCounter++;
                line = new StringBuffer();
            } else {
                line.append(level[i]);
                line.append(',');
            }
        }
        return allLines;
    }

    public static void main(String[] args) {

        long genSeed = System.currentTimeMillis();
        LevelGenerator gen = new LevelGenerator(genSeed);
        gen.init(11, 3, 4, 0.5, new Types.TRIBE[]{XIN_XI, OUMAJI});
        gen.generate();
        gen.toCSV("levels/levelgen_test.csv");
        gen.print();
        
    }
}
