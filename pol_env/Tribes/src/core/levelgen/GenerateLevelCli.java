package core.levelgen;

import core.Types;

import java.util.ArrayList;
import java.util.List;

/**
 * Small CLI wrapper around LevelGenerator so maps can be generated and frozen as CSV files.
 *
 * Usage:
 *   java -cp out;lib/json.jar core.levelgen.GenerateLevelCli <seed> <mapSize> <outputCsv> <tribe1> [tribe2...]
 *
 * Example:
 *   java -cp out;lib/json.jar core.levelgen.GenerateLevelCli 12345 16 levels/phase1_fixed.csv BARDUR XIN_XI
 */
public class GenerateLevelCli {

    public static void main(String[] args) {
        if (args.length < 5) {
            System.out.println("Usage: <seed> <mapSize> <outputCsv> <tribe1> [tribe2...]");
            System.exit(1);
        }

        long seed = Long.parseLong(args[0]);
        int mapSize = Integer.parseInt(args[1]);
        String outputCsv = args[2];

        List<Types.TRIBE> tribeList = new ArrayList<>();
        for (int i = 3; i < args.length; i++) {
            tribeList.add(parseTribe(args[i]));
        }
        Types.TRIBE[] tribes = tribeList.toArray(new Types.TRIBE[0]);

        LevelGenerator gen = new LevelGenerator(seed);
        gen.init(mapSize, 3, 4, 0.5, tribes);
        gen.generate();
        gen.toCSV(outputCsv);
        System.out.println("Generated map:");
        gen.print();
        System.out.println("Saved CSV to: " + outputCsv);
    }

    private static Types.TRIBE parseTribe(String raw) {
        String key = raw.trim().toUpperCase().replace("-", "_").replace(" ", "_");
        switch (key) {
            case "XIN_XI": return Types.TRIBE.XIN_XI;
            case "IMPERIUS": return Types.TRIBE.IMPERIUS;
            case "BARDUR": return Types.TRIBE.BARDUR;
            case "OUMAJI": return Types.TRIBE.OUMAJI;
            case "KICKOO": return Types.TRIBE.KICKOO;
            case "HOODRICK": return Types.TRIBE.HOODRICK;
            case "LUXIDOOR": return Types.TRIBE.LUXIDOOR;
            case "VENGIR": return Types.TRIBE.VENGIR;
            case "ZEBASI": return Types.TRIBE.ZEBASI;
            case "AI_MO": return Types.TRIBE.AI_MO;
            case "QUETZALI": return Types.TRIBE.QUETZALI;
            case "YADAKK": return Types.TRIBE.YADAKK;
            default:
                throw new IllegalArgumentException("Unknown tribe: " + raw);
        }
    }
}

