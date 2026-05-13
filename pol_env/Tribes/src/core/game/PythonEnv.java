package core.game;

import core.Types;
import core.actions.Action;
import core.actions.cityactions.Build;
import core.actions.cityactions.ClearForest;
import core.actions.cityactions.CityAction;
import core.actions.cityactions.GrowForest;
import core.actions.cityactions.LevelUp;
import core.actions.cityactions.ResourceGathering;
import core.actions.cityactions.Spawn;
import core.actions.tribeactions.ResearchTech;
import core.actions.unitactions.Capture;
import core.actions.unitactions.Examine;
import core.actions.unitactions.Move;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Battleship;
import core.actors.units.Boat;
import core.actors.units.Ship;
import core.actors.units.Unit;
import org.json.JSONArray;
import org.json.JSONObject;
import gui.GUI;
import gui.WindowInput;
import players.ActionController;
import players.KeyController;
import players.Agent;
import players.DoNothingAgent;

import java.util.ArrayList;
import java.util.Random;

/**
 * Minimal bridge for Python interop. Exposes init, list-actions, step-by-index, observation JSON, and status.
 */
public class PythonEnv {

    private GameState gs;
    private Random rnd;
    private GUI gui;
    private Game viewerGame;
    private boolean soloNoOpponentMode = false;

    /**
     * Initialize from a CSV level file path (relative to CWD or absolute), seed, and game mode.
     */
    public void initFromLevel(String filename, long seed, Types.GAME_MODE gameMode) {
        this.rnd = new Random(seed);
        this.gs = new GameState(rnd, gameMode);
        this.gs.init(filename);

        // Start the turn for tribe 0, mirroring Game.processTurn preconditions
        Tribe t0 = gs.getTribe(0);
        gs.initTurn(t0);
        gs.computePlayerActions(t0);
    }

    public int getActiveTribeID() {
        return gs.getActiveTribeID();
    }

    public void setActiveTribeID(int tribeId) {
        gs.getBoard().setActiveTribeID(tribeId);
        // Recompute actions for the new active tribe
        Tribe newActiveTribe = gs.getTribe(tribeId);
        gs.computePlayerActions(newActiveTribe);
    }

    public int getTick() {
        return gs.getTick();
    }

    /**
     * Enables/disables single-tribe continuation mode.
     * When enabled and a map has exactly one tribe, END_TURN will keep the game
     * advancing instead of stopping at immediate SCORE-mode game-over.
     */
    public void setSoloNoOpponentMode(boolean enabled) {
        this.soloNoOpponentMode = enabled;
    }

    public boolean getSoloNoOpponentMode() {
        return soloNoOpponentMode;
    }

    public boolean isDone() {
        return gs.isGameOver();
    }

    public String getGameMode() {
        return gs.getGameMode().toString();
    }

    public int[] getScores() {
        Tribe[] tribes = gs.getTribes();
        int[] scores = new int[tribes.length];
        for (int i = 0; i < tribes.length; i++) scores[i] = tribes[i].getScore();
        return scores;
    }

    /**
     * Returns each available action encoded as a JSON string: {idx, type, repr}
     */
    public java.util.List<String> listActionsJson() {
        ArrayList<Action> acts = gs.getAllAvailableActions();
        ArrayList<String> out = new ArrayList<>(acts.size());
        for (int i = 0; i < acts.size(); i++) {
            JSONObject jo = buildActionJsonObject(acts.get(i), i);
            out.add(jo.toString());
        }
        return out;
    }

    /**
     * Returns all available actions encoded as one JSON array string.
     * Each array element is the same action-object payload used by listActionsJson(),
     * preserving the exact order.
     */
    public String listActionsJsonBatch() {
        ArrayList<Action> acts = gs.getAllAvailableActions();
        JSONArray arr = new JSONArray();
        for (int i = 0; i < acts.size(); i++) {
            arr.put(buildActionJsonObject(acts.get(i), i));
        }
        return arr.toString();
    }

    private JSONObject buildActionJsonObject(Action a, int idx) {
        JSONObject jo = new JSONObject();
        jo.put("idx", idx);
        jo.put("type", a.getActionType().toString());
        jo.put("repr", a.toString());
        jo.put("schema_version", 1);
        addStructuredActionFields(jo, a);
        return jo;
    }

    private void addStructuredActionFields(JSONObject jo, Action a) {
        if (a == null) return;

        if (a instanceof Move) {
            Move m = (Move) a;
            jo.put("unit_id", m.getUnitId());
            Unit u = safeGetUnit(m.getUnitId());
            if (u != null) {
                jo.put("src_x", u.getPosition().x);
                jo.put("src_y", u.getPosition().y);
            }
            if (m.getDestination() != null) {
                jo.put("dst_x", m.getDestination().x);
                jo.put("dst_y", m.getDestination().y);
            }
            return;
        }

        if (a instanceof Capture) {
            Capture c = (Capture) a;
            jo.put("unit_id", c.getUnitId());
            jo.put("capture_type", c.getCaptureType() != null ? c.getCaptureType().toString() : JSONObject.NULL);
            jo.put("target_city_id", c.getTargetCity());
            Unit u = safeGetUnit(c.getUnitId());
            if (u != null) {
                jo.put("src_x", u.getPosition().x);
                jo.put("src_y", u.getPosition().y);
            }
            City city = safeGetCity(c.getTargetCity());
            if (city != null) {
                jo.put("target_x", city.getPosition().x);
                jo.put("target_y", city.getPosition().y);
                jo.put("target_city_tile", city.getPosition().x * gs.getBoard().getSize() + city.getPosition().y);
            } else if (u != null) {
                // Village capture target is current unit tile.
                jo.put("target_x", u.getPosition().x);
                jo.put("target_y", u.getPosition().y);
                jo.put("target_city_tile", u.getPosition().x * gs.getBoard().getSize() + u.getPosition().y);
            }
            return;
        }

        if (a instanceof Spawn) {
            Spawn s = (Spawn) a;
            addCityFields(jo, s);
            jo.put("unit_type", s.getUnitType() != null ? s.getUnitType().toString() : JSONObject.NULL);
            return;
        }

        if (a instanceof ResourceGathering) {
            ResourceGathering rg = (ResourceGathering) a;
            addCityFields(jo, rg);
            jo.put("resource_type", rg.getResource() != null ? rg.getResource().toString() : JSONObject.NULL);
            putTargetFieldsFromCityAction(jo, rg);
            return;
        }

        if (a instanceof Build) {
            Build b = (Build) a;
            addCityFields(jo, b);
            jo.put("building_type", b.getBuildingType() != null ? b.getBuildingType().toString() : JSONObject.NULL);
            putTargetFieldsFromCityAction(jo, b);
            return;
        }

        if (a instanceof ClearForest) {
            ClearForest cf = (ClearForest) a;
            addCityFields(jo, cf);
            putTargetFieldsFromCityAction(jo, cf);
            return;
        }

        if (a instanceof GrowForest) {
            GrowForest gf = (GrowForest) a;
            addCityFields(jo, gf);
            putTargetFieldsFromCityAction(jo, gf);
            return;
        }

        if (a instanceof LevelUp) {
            LevelUp lu = (LevelUp) a;
            addCityFields(jo, lu);
            jo.put("levelup_choice", lu.getBonus() != null ? lu.getBonus().toString() : JSONObject.NULL);
            putTargetFieldsFromCityAction(jo, lu);
            return;
        }

        if (a instanceof ResearchTech) {
            ResearchTech rt = (ResearchTech) a;
            jo.put("tribe_id", rt.getTribeId());
            jo.put("tech_type", rt.getTech() != null ? rt.getTech().toString() : JSONObject.NULL);
            return;
        }

        if (a instanceof Examine) {
            Examine ex = (Examine) a;
            jo.put("unit_id", ex.getUnitId());
            Unit u = safeGetUnit(ex.getUnitId());
            if (u != null) {
                jo.put("src_x", u.getPosition().x);
                jo.put("src_y", u.getPosition().y);
            }
        }
    }

    private void addCityFields(JSONObject jo, CityAction action) {
        int cityId = action.getCityId();
        jo.put("city_id", cityId);
        City city = safeGetCity(cityId);
        if (city != null) {
            jo.put("city_x", city.getPosition().x);
            jo.put("city_y", city.getPosition().y);
            jo.put("city_tile", city.getPosition().x * gs.getBoard().getSize() + city.getPosition().y);
        }
    }

    private void putTargetFieldsFromCityAction(JSONObject jo, CityAction action) {
        if (action.getTargetPos() == null) return;
        int tx = action.getTargetPos().x;
        int ty = action.getTargetPos().y;
        jo.put("target_x", tx);
        jo.put("target_y", ty);
        jo.put("target_tile", tx * gs.getBoard().getSize() + ty);
    }

    private Unit safeGetUnit(int unitId) {
        try {
            Object actor = gs.getActor(unitId);
            if (actor instanceof Unit) return (Unit) actor;
        } catch (Exception ignored) {}
        return null;
    }

    private City safeGetCity(int cityId) {
        try {
            Object actor = gs.getActor(cityId);
            if (actor instanceof City) return (City) actor;
        } catch (Exception ignored) {}
        return null;
    }

    public int actionCount() {
        return gs.getAllAvailableActions().size();
    }

    /**
     * Applies the action by its index in the last listActionsJson() call and recomputes actions.
     */
    public void stepByIndex(int idx) {
        ArrayList<Action> acts = gs.getAllAvailableActions();
        if (idx < 0 || idx >= acts.size()) throw new IllegalArgumentException("Invalid action index: " + idx);
        Action selected = acts.get(idx);
        gs.advance(selected, true);

        // In pure single-tribe maps, SCORE mode ends immediately after END_TURN.
        // For wrapper-driven T10 training, keep turns progressing when explicitly requested.
        boolean singleTribe = gs.getTribes() != null && gs.getTribes().length == 1;
        boolean selectedEndTurn = selected != null && selected.getActionType() == Types.ACTION.END_TURN;
        if (soloNoOpponentMode && singleTribe && selectedEndTurn) {
            gs.setGameIsOver(false);
            gs.invalidateComputedActions();
            Tribe active = gs.getActiveTribe();
            if (active != null) {
                gs.initTurn(active);
                gs.computePlayerActions(active);
            }
        }
    }

    private String observationJsonFromState(GameState state) {
        if (state == null) {
            return "{}";
        }

        JSONObject game = new JSONObject();

        // Board INFO (2D arrays)
        JSONObject board = new JSONObject();
        JSONArray terrain2D = new JSONArray();
        JSONArray resource2D = new JSONArray();
        JSONArray unit2D = new JSONArray();
        JSONArray city2D = new JSONArray();
        JSONArray building2D = new JSONArray();
        JSONArray network2D = new JSONArray();

        Board b = state.getBoard();
        for (int i = 0; i < b.getSize(); i++) {
            JSONArray terrain = new JSONArray();
            JSONArray resource = new JSONArray();
            JSONArray units = new JSONArray();
            JSONArray cities = new JSONArray();
            JSONArray buildings = new JSONArray();
            JSONArray networks = new JSONArray();

            for (int j = 0; j < b.getSize(); j++) {
                terrain.put(b.getTerrainAt(i, j).getKey());
                resource.put(b.getResourceAt(i, j) != null ? b.getResourceAt(i, j).getKey() : -1);
                units.put(b.getUnitIDAt(i, j));
                cities.put(b.getCityIdAt(i, j));
                buildings.put(b.getBuildingAt(i, j) != null ? b.getBuildingAt(i, j).getKey() : -1);
                networks.put(b.getNetworkTilesAt(i, j));
            }

            terrain2D.put(terrain);
            resource2D.put(resource);
            unit2D.put(units);
            city2D.put(cities);
            building2D.put(buildings);
            network2D.put(networks);
        }

        // Unit INFO
        JSONObject unit = new JSONObject();
        for (Unit u : getAllUnits(b)) {
            JSONObject uInfo = new JSONObject();
            uInfo.put("type", u.getType().getKey());
            if (u.getType() == Types.UNIT.BOAT) {
                uInfo.put("baseLandType", ((Boat) u).getBaseLandUnit().getKey());
            } else if (u.getType() == Types.UNIT.SHIP) {
                uInfo.put("baseLandType", ((Ship) u).getBaseLandUnit().getKey());
            } else if (u.getType() == Types.UNIT.BATTLESHIP) {
                uInfo.put("baseLandType", ((Battleship) u).getBaseLandUnit().getKey());
            }
            uInfo.put("x", u.getPosition().x);
            uInfo.put("y", u.getPosition().y);
            uInfo.put("kill", u.getKills());
            uInfo.put("isVeteran", u.isVeteran());
            uInfo.put("cityID", u.getCityId());
            uInfo.put("tribeId", u.getTribeId());
            uInfo.put("currentHP", u.getCurrentHP());
            unit.put(String.valueOf(u.getActorId()), uInfo);
        }

        // City INFO
        JSONObject city = new JSONObject();
        for (City c : getAllCities(b)) {
            JSONObject cInfo = new JSONObject();
            cInfo.put("x", c.getPosition().x);
            cInfo.put("y", c.getPosition().y);
            cInfo.put("tribeID", c.getTribeId());
            cInfo.put("population_need", c.getPopulation_need());
            cInfo.put("bound", c.getBound());
            cInfo.put("level", c.getLevel());
            cInfo.put("isCapital", c.isCapital());
            cInfo.put("population", c.getPopulation());
            cInfo.put("production", c.getProduction());
            cInfo.put("hasWalls", c.hasWalls());
            cInfo.put("pointsWorth", c.getPointsWorth());
            JSONArray buildingList = new JSONArray();
            for (core.actors.Building bld : c.getBuildings()) {
                JSONObject bInfo = new JSONObject();
                bInfo.put("x", bld.position.x);
                bInfo.put("y", bld.position.y);
                bInfo.put("type", bld.type.getKey());
                if (bld.type.isTemple()) {
                    core.actors.Temple t = (core.actors.Temple) bld;
                    bInfo.put("level", t.getLevel());
                    bInfo.put("turnsToScore", t.getTurnsToScore());
                }
                buildingList.put(bInfo);
            }
            cInfo.put("buildings", buildingList);
            cInfo.put("units", c.getUnitsID());
            city.put(String.valueOf(c.getActorId()), cInfo);
        }

        // Tribes INFO (subset)
        JSONObject tribesINFO = new JSONObject();
        Tribe[] tribes = state.getTribes();
        for (Tribe t : tribes) {
            JSONObject tribeInfo = new JSONObject();
            tribeInfo.put("citiesID", t.getCitiesID());
            tribeInfo.put("capitalID", t.getCapitalID());
            tribeInfo.put("type", t.getType().getKey());
            tribeInfo.put("star", t.getStars());
            tribeInfo.put("winner", t.getWinner().getKey());
            tribeInfo.put("score", t.getScore());
            tribeInfo.put("extraUnits", t.getExtraUnits());
            tribeInfo.put("nKills", t.getnKills());
            tribeInfo.put("nPacifistCount", t.getnPacifistCount());
            tribesINFO.put(String.valueOf(t.getActorId()), tribeInfo);
        }

        board.put("terrain", terrain2D);
        board.put("resource", resource2D);
        board.put("unitID", unit2D);
        board.put("cityID", city2D);
        board.put("network", network2D);
        board.put("building", building2D);
        board.put("actorIDcounter", b.getActorIDcounter());

        game.put("board", board);
        game.put("unit", unit);
        game.put("city", city);
        game.put("tribes", tribesINFO);
        game.put("tick", state.getTick());
        game.put("gameIsOver", state.isGameOver());
        game.put("activeTribeID", state.getActiveTribeID());
        game.put("gameMode", state.getGameMode().getKey());

        return game.toString();
    }

    /**
     * Returns an observation JSON similar to GameSaver output but in-memory.
     * This variant applies active-player fog-of-war.
     */
    public String observationJson() {
        int povPlayer = gs.getActiveTribeID();
        GameState pov = gs.copy(povPlayer);
        return observationJsonFromState(pov);
    }

    /**
     * Diagnostic-only full-visibility observation built from runtime state.
     * This bypasses POV fog and should not be used for training.
     */
    public String observationJsonFull() {
        return observationJsonFromState(gs);
    }

    /**
     * Opens a Java GUI window using the existing viewer components. Safe to call multiple times.
     */
    public void openGui() {
        if (gui != null) return;

        // Build a minimal Game just for GUI plumbing (players, pause state, etc.)
        viewerGame = new Game();
        int n = gs.getTribes().length;
        java.util.ArrayList<Agent> players = new java.util.ArrayList<>();
        long agentSeed = 0L;
        for (int i = 0; i < n; i++) players.add(new DoNothingAgent(agentSeed));

        // Initialize viewerGame with a generated level matching current tribes to set up players array
        // We won't use viewerGame's internal state for rendering; GUI.update uses the GameState we pass.
        Types.TRIBE[] tribes = new Types.TRIBE[n];
        for (int i = 0; i < n; i++) tribes[i] = gs.getTribe(i).getType();
        long levelSeed = 0L;
        viewerGame.init(players, levelSeed, tribes, 0L, gs.getGameMode());

        KeyController ki = new KeyController(true);
        WindowInput wi = new WindowInput();
        ActionController ac = new ActionController();
        gui = new GUI(viewerGame, "Tribes Viewer", ki, wi, ac, false);

        // Force zoomed out so the whole board fits initially
        try {
            int boardSize = gs.getBoard().getSize();
            int viewSize = core.Constants.GUI_GAME_VIEW_SIZE > 0 ? core.Constants.GUI_GAME_VIEW_SIZE : 600;
            int cell = Math.max(2, viewSize / Math.max(1, boardSize));
            core.Constants.CELL_SIZE = cell;
            gui.repaint();
        } catch (Exception ignored) {}
    }

    /**
     * Renders the current GameState in the Java GUI window; call openGui() beforehand.
     */
    public void renderGui() {
        if (gui == null) openGui();
        // Render from the active tribe POV so fog-of-war is visible in the viewer.
        gui.update(gs.copy(gs.getActiveTribeID()), null);
    }

    /**
     * Closes the Java GUI window if open.
     */
    public void closeGui() {
        if (gui != null) {
            gui.dispose();
            gui = null;
            viewerGame = null;
        }
    }

    private static ArrayList<City> getAllCities(Board board) {
        Tribe[] tribes = board.getTribes();
        ArrayList<City> cityActors = new ArrayList<>();
        for (Tribe t : tribes) {
            ArrayList<Integer> cities = t.getCitiesID();
            for (Integer cityId : cities) {
                cityActors.add((City) board.getActor(cityId));
            }
        }
        return cityActors;
    }

    private static ArrayList<Unit> getAllUnits(Board board) {
        Tribe[] tribes = board.getTribes();
        ArrayList<Unit> unitActors = new ArrayList<>();
        for (Tribe t : tribes) {
            ArrayList<Integer> cities = t.getCitiesID();
            for (Integer cityId : cities) {
                City c = (City) board.getActor(cityId);
                for (Integer unitId : c.getUnitsID()) {
                    Unit unit = (Unit) board.getActor(unitId);
                    unitActors.add(unit);
                }
            }
            for (Integer unitId : t.getExtraUnits()) {
                Unit unit = (Unit) board.getActor(unitId);
                unitActors.add(unit);
            }
        }
        return unitActors;
    }
}


