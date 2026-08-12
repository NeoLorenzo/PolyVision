package core.levelgen;

import core.TribesConfig;
import core.Types;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public final class LevelGeneratorRegressionTest {
    private static final int MAP_SIZE = 12;
    private static final double EPSILON = 1e-12;

    private LevelGeneratorRegressionTest() {}

    public static void main(String[] args) {
        testBardurRuntimeConfiguration();
        testBardurStartingAnimalGuarantee();
        testRuinCountAndSpacing(0L);
        testRuinCountAndSpacing(7L);
        System.out.println("LevelGeneratorRegressionTest: PASS");
    }

    private static void testBardurRuntimeConfiguration() {
        LevelGenerator generator = new LevelGenerator(0L);
        assertDoubleEquals(0.8, generator.getTribeProb("FOREST", Types.TRIBE.BARDUR), "Bardur FOREST");
        assertDoubleEquals(1.0, generator.getTribeProb("MOUNTAIN", Types.TRIBE.BARDUR), "Bardur MOUNTAIN");
        assertDoubleEquals(1.0, generator.getTribeProb("FRUIT", Types.TRIBE.BARDUR), "Bardur FRUIT");
        assertDoubleEquals(0.0, generator.getTribeProb("CROPS", Types.TRIBE.BARDUR), "Bardur CROPS");
        assertDoubleEquals(1.0, generator.getTribeProb("ANIMAL", Types.TRIBE.BARDUR), "Bardur ANIMAL");
        assertDoubleEquals(1.0, generator.getTribeProb("ORE", Types.TRIBE.BARDUR), "Bardur ORE");
    }

    private static void testBardurStartingAnimalGuarantee() {
        LevelGenerator generator = generateBardurDrylands(0L);
        int capital = findBardurCapital(generator);
        assertTrue(generator.checkResources(Types.RESOURCE.ANIMAL.getMapChar(), capital) >= 2,
                "Generated Bardur starting area should contain at least two animals");

        for (int neighbour : generator.circle(capital, 1)) {
            generator.writeTile(neighbour, null, "");
        }
        assertEquals(0, generator.checkResources(Types.RESOURCE.ANIMAL.getMapChar(), capital),
                "cleared Bardur starting animals");
        generator.postGenerate(
                Types.RESOURCE.ANIMAL.getMapChar(),
                Types.TERRAIN.FOREST.getMapChar(),
                2,
                capital
        );
        assertTrue(generator.checkResources(Types.RESOURCE.ANIMAL.getMapChar(), capital) >= 2,
                "Bardur starting-animal guarantee should be independent of the global multiplier");
    }

    private static void testRuinCountAndSpacing(long seed) {
        LevelGenerator generator = generateBardurDrylands(seed);
        ArrayList<Integer> ruins = new ArrayList<>();
        String ruinResource = String.valueOf(Types.RESOURCE.RUINS.getMapChar());
        for (int tile = 0; tile < MAP_SIZE * MAP_SIZE; tile++) {
            if (ruinResource.equals(generator.getResource(tile))) {
                ruins.add(tile);
            }
        }

        int expected = LevelGenerator.ruinTargetForMapSize(MAP_SIZE);
        Set<Integer> uniqueRuins = new HashSet<>(ruins);
        assertEquals(expected, ruins.size(), "seed " + seed + " ruin tile count");
        assertEquals(expected, uniqueRuins.size(), "seed " + seed + " unique ruin tile count");

        for (int i = 0; i < ruins.size(); i++) {
            for (int j = i + 1; j < ruins.size(); j++) {
                int distance = generator.distance(ruins.get(i), ruins.get(j), MAP_SIZE);
                assertTrue(distance > 1,
                        "seed " + seed + " ruins must not be adjacent: " + ruins.get(i) + " and " + ruins.get(j));
            }
        }
    }

    private static LevelGenerator generateBardurDrylands(long seed) {
        LevelGenerator generator = new LevelGenerator(seed);
        generator.init(
                MAP_SIZE,
                TribesConfig.MAP_TYPE.DRYLANDS.getSmoothing(),
                TribesConfig.MAP_TYPE.DRYLANDS.getRelief(),
                1.0,
                new Types.TRIBE[]{Types.TRIBE.BARDUR},
                TribesConfig.MAP_TYPE.DRYLANDS
        );
        generator.generate();
        return generator;
    }

    private static int findBardurCapital(LevelGenerator generator) {
        String bardurKey = String.valueOf(Types.TRIBE.BARDUR.getKey());
        for (int tile = 0; tile < MAP_SIZE * MAP_SIZE; tile++) {
            if (generator.getTerrain(tile) == Types.TERRAIN.CITY.getMapChar()
                    && bardurKey.equals(generator.getResource(tile))) {
                return tile;
            }
        }
        throw new AssertionError("Bardur capital not found");
    }

    private static void assertDoubleEquals(double expected, double actual, String label) {
        if (Math.abs(expected - actual) > EPSILON) {
            throw new AssertionError(label + ": expected " + expected + ", got " + actual);
        }
    }

    private static void assertEquals(int expected, int actual, String label) {
        if (expected != actual) {
            throw new AssertionError(label + ": expected " + expected + ", got " + actual);
        }
    }

    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
