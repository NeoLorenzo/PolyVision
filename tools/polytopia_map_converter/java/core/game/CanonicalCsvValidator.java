package core.game;

import core.Types;
import org.json.JSONArray;
import org.json.JSONObject;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/** Independent semantic validator for canonical JSON -> CSV -> LevelLoader. */
public final class CanonicalCsvValidator {
    private CanonicalCsvValidator() {}

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: CanonicalCsvValidator <canonical-json-dir> <csv-dir>");
            System.exit(2);
        }
        Path jsonDir = Paths.get(args[0]);
        Path csvDir = Paths.get(args[1]);
        List<Path> inputs = new ArrayList<>();
        try (java.util.stream.Stream<Path> stream = Files.list(jsonDir)) {
            stream.filter(path -> path.getFileName().toString().endsWith(".json"))
                    .sorted(Comparator.comparing(path -> path.getFileName().toString()))
                    .forEach(inputs::add);
        }
        int loaded = 0;
        for (Path jsonPath : inputs) {
            String basename = jsonPath.getFileName().toString().replaceFirst("\\.json$", "");
            Path csvPath = csvDir.resolve(basename + ".csv");
            JSONObject canonical = new JSONObject(Files.readString(jsonPath, StandardCharsets.UTF_8));
            String[] lines = Files.readAllLines(csvPath, StandardCharsets.UTF_8).toArray(new String[0]);
            Board board = new LevelLoader().buildLevel(lines, new Random(0L));
            validateMap(canonical, board, jsonPath.getFileName().toString());
            loaded++;
        }
        System.out.println("Java LevelLoader semantic validation:");
        System.out.println("  CSV maps tested: " + inputs.size());
        System.out.println("  Loaded:          " + loaded);
        System.out.println("  Failed:          0");
    }

    private static void validateMap(JSONObject map, Board board, String name) {
        if (map.getInt("schema_version") != 1) fail(name, "unsupported schema");
        JSONObject game = map.getJSONObject("game");
        int width = game.getInt("map_width");
        int height = game.getInt("map_height");
        if (width != height || board.getSize() != width) fail(name, "board dimensions differ from canonical JSON");
        if (board.getTribes().length != 1 || board.getTribes()[0].getType() != Types.TRIBE.BARDUR) {
            fail(name, "expected exactly one Bardur tribe");
        }
        JSONArray tiles = map.getJSONArray("tiles");
        if (tiles.length() != width * height) fail(name, "canonical tile count mismatch");
        int capitals = 0;
        for (int index = 0; index < tiles.length(); index++) {
            JSONObject tile = tiles.getJSONObject(index);
            int x = tile.getInt("x");
            int y = tile.getInt("y");
            Types.TERRAIN expectedTerrain = expectedTerrain(tile);
            Types.RESOURCE expectedResource = expectedResource(tile);
            // LevelLoader stores CSV row i first and column j second. Canonical
            // CSV rows are y and columns are x, so the board lookup is (y, x).
            if (board.getTerrainAt(y, x) != expectedTerrain) {
                fail(name, "terrain mismatch at canonical (" + x + "," + y + ")");
            }
            if (board.getResourceAt(y, x) != expectedResource) {
                fail(name, "resource mismatch at canonical (" + x + "," + y + ")");
            }
            if (tile.getBoolean("capital")) capitals++;
        }
        if (capitals != 1 || board.getCapitalIDs().length != 1 || board.getCapitalIDs()[0] < 0) {
            fail(name, "expected one loaded capital");
        }
    }

    private static Types.TERRAIN expectedTerrain(JSONObject tile) {
        if (tile.getBoolean("capital")) return Types.TERRAIN.CITY;
        if (!tile.isNull("improvement") && tile.getJSONObject("improvement").getInt("id") == 1) return Types.TERRAIN.VILLAGE;
        switch (tile.getInt("terrain_id")) {
            case 1: return Types.TERRAIN.SHALLOW_WATER;
            case 2: return Types.TERRAIN.DEEP_WATER;
            case 3: return Types.TERRAIN.PLAIN;
            case 4: return Types.TERRAIN.MOUNTAIN;
            case 5: return Types.TERRAIN.FOREST;
            default: throw new IllegalArgumentException("unsupported terrain ID");
        }
    }

    private static Types.RESOURCE expectedResource(JSONObject tile) {
        if (!tile.isNull("improvement") && tile.getJSONObject("improvement").getInt("id") == 2) return Types.RESOURCE.RUINS;
        if (tile.isNull("resource")) return null;
        switch (tile.getJSONObject("resource").getInt("id")) {
            case 1: return Types.RESOURCE.ANIMAL;
            case 2: return Types.RESOURCE.CROPS;
            case 3: return Types.RESOURCE.FISH;
            case 4: return Types.RESOURCE.WHALES;
            case 5: return Types.RESOURCE.ORE;
            case 6: return Types.RESOURCE.FRUIT;
            default: throw new IllegalArgumentException("unsupported resource ID");
        }
    }

    private static void fail(String name, String reason) {
        throw new IllegalStateException(name + ": " + reason);
    }
}

