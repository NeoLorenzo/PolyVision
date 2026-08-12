package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"

	polytopia "github.com/samuelyuan/polytopiamapmodelgo"
)

const schemaVersion = 1

type CanonicalMap struct {
	SchemaVersion int               `json:"schema_version"`
	Source        SourceIdentity    `json:"source"`
	MapSHA256     string            `json:"map_sha256"`
	Game          GameMetadata      `json:"game"`
	Players       []CanonicalPlayer `json:"players"`
	Tiles         []CanonicalTile   `json:"tiles"`
}

type SourceIdentity struct {
	Filename  string `json:"filename"`
	SHA256    string `json:"sha256"`
	SizeBytes int64  `json:"size_bytes"`
}

type GameMetadata struct {
	GameVersion       int         `json:"game_version"`
	Seed              int32       `json:"seed"`
	MapName           string      `json:"map_name"`
	MapWidth          int         `json:"map_width"`
	MapHeight         int         `json:"map_height"`
	MapSquareSize     int         `json:"map_square_size"`
	MapPreset         int         `json:"map_preset"`
	GameType          int         `json:"game_type"`
	GameDifficulty    int         `json:"game_difficulty"`
	NumOpponents      int         `json:"num_opponents"`
	GameModeBase      uint8       `json:"game_mode_base"`
	GameModeRules     uint8       `json:"game_mode_rules"`
	DisabledTribeIDs  []int       `json:"disabled_tribe_ids"`
	UnlockedTribeIDs  []int       `json:"unlocked_tribe_ids"`
	SelectedTribeSkin []TribeSkin `json:"selected_tribe_skins"`
}

type TribeSkin struct {
	TribeID int `json:"tribe_id"`
	SkinID  int `json:"skin_id"`
}

type CanonicalPlayer struct {
	PlayerID               int   `json:"player_id"`
	TribeID                int   `json:"tribe_id"`
	StartX                 int   `json:"start_x"`
	StartY                 int   `json:"start_y"`
	Currency               int   `json:"currency"`
	Score                  int   `json:"score"`
	NumCities              int   `json:"num_cities"`
	AvailableTechnologyIDs []int `json:"available_technology_ids"`
	PlayerSkinID           int   `json:"player_skin_id"`
}

type CanonicalTile struct {
	X             int                   `json:"x"`
	Y             int                   `json:"y"`
	TerrainID     int                   `json:"terrain_id"`
	ClimateID     int                   `json:"climate_id"`
	Altitude      int                   `json:"altitude"`
	OwnerID       int                   `json:"owner_id"`
	Capital       bool                  `json:"capital"`
	CapitalX      int                   `json:"capital_x"`
	CapitalY      int                   `json:"capital_y"`
	Resource      *CanonicalResource    `json:"resource"`
	Improvement   *CanonicalImprovement `json:"improvement"`
	HasRoad       bool                  `json:"has_road"`
	HasWaterRoute bool                  `json:"has_water_route"`
	TileSkin      int                   `json:"tile_skin"`
	Flooded       bool                  `json:"flooded"`
	FloodedValue  int                   `json:"flooded_value"`
}

type CanonicalResource struct {
	ID int `json:"id"`
}

type CanonicalImprovement struct {
	ID                     int    `json:"id"`
	Level                  int    `json:"level"`
	FoundedTurn            int    `json:"founded_turn"`
	CurrentPopulation      int    `json:"current_population"`
	TotalPopulation        int    `json:"total_population"`
	Production             int    `json:"production"`
	BaseScore              int    `json:"base_score"`
	BorderSize             int    `json:"border_size"`
	UpgradeCount           int    `json:"upgrade_count"`
	ConnectedPlayerCapital int    `json:"connected_player_capital"`
	CityName               string `json:"city_name"`
	FoundedTribeID         int    `json:"founded_tribe_id"`
	CityRewardIDs          []int  `json:"city_reward_ids"`
	RebellionFlag          int    `json:"rebellion_flag"`
	RebellionBuffer        []int  `json:"rebellion_buffer"`
}

// canonicalizeSave deliberately reads InitialTileData and InitialPlayerData.
// Current TileData and PlayerData are never accepted by the canonical layer.
func canonicalizeSave(save *polytopia.PolytopiaSaveOutput, source SourceIdentity) (*CanonicalMap, error) {
	if save == nil {
		return nil, fmt.Errorf("parser returned a nil save")
	}
	if save.GameVersion <= 0 {
		return nil, fmt.Errorf("game version is not readable: %d", save.GameVersion)
	}

	header := save.MapHeaderOutput
	width, height := header.MapWidth, header.MapHeight
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid map dimensions %dx%d", width, height)
	}
	if len(save.InitialPlayerData) == 0 {
		return nil, fmt.Errorf("InitialPlayerData is empty")
	}
	players := canonicalizePlayers(save.InitialPlayerData)
	tiles, err := canonicalizeInitialTileRows(width, height, save.InitialTileData)
	if err != nil {
		return nil, err
	}

	skins := make([]TribeSkin, len(header.SelectedTribeSkins))
	for i, skin := range header.SelectedTribeSkins {
		skins[i] = TribeSkin{TribeID: skin.Tribe, SkinID: skin.Skin}
	}
	sort.Slice(skins, func(i, j int) bool {
		if skins[i].TribeID != skins[j].TribeID {
			return skins[i].TribeID < skins[j].TribeID
		}
		return skins[i].SkinID < skins[j].SkinID
	})

	result := &CanonicalMap{
		SchemaVersion: schemaVersion,
		Source:        source,
		Game: GameMetadata{
			GameVersion:       save.GameVersion,
			Seed:              header.MapHeaderInput.Seed,
			MapName:           header.MapName,
			MapWidth:          width,
			MapHeight:         height,
			MapSquareSize:     header.MapSquareSize,
			MapPreset:         header.MapPreset,
			GameType:          header.GameType,
			GameDifficulty:    header.GameDifficulty,
			NumOpponents:      header.NumOpponents,
			GameModeBase:      header.MapHeaderInput.GameModeBase,
			GameModeRules:     header.MapHeaderInput.GameModeRules,
			DisabledTribeIDs:  nonNilInts(header.DisabledTribesArr),
			UnlockedTribeIDs:  nonNilInts(header.UnlockedTribesArr),
			SelectedTribeSkin: skins,
		},
		Players: players,
		Tiles:   tiles,
	}
	result.MapSHA256, err = calculateMapSHA256(result)
	if err != nil {
		return nil, err
	}
	return result, nil
}

func canonicalizePlayers(input []polytopia.PlayerData) []CanonicalPlayer {
	players := make([]CanonicalPlayer, len(input))
	for i, player := range input {
		players[i] = CanonicalPlayer{
			PlayerID:               player.PlayerId,
			TribeID:                player.Tribe,
			StartX:                 player.StartTileCoordinates[0],
			StartY:                 player.StartTileCoordinates[1],
			Currency:               player.Currency,
			Score:                  player.Score,
			NumCities:              player.NumCities,
			AvailableTechnologyIDs: nonNilInts(player.AvailableTech),
			PlayerSkinID:           player.PlayerSkin,
		}
		sort.Ints(players[i].AvailableTechnologyIDs)
	}
	sort.Slice(players, func(i, j int) bool { return players[i].PlayerID < players[j].PlayerID })
	return players
}

func canonicalizeInitialTileRows(width, height int, rows [][]polytopia.TileData) ([]CanonicalTile, error) {
	if len(rows) != height {
		return nil, fmt.Errorf("expected %d InitialTileData rows but parsed %d", height, len(rows))
	}
	flat := make([]polytopia.TileData, 0, width*height)
	for y, row := range rows {
		if len(row) != width {
			return nil, fmt.Errorf("InitialTileData row %d: expected %d tiles but parsed %d", y, width, len(row))
		}
		flat = append(flat, row...)
	}
	return canonicalizeTiles(width, height, flat)
}

func canonicalizeTiles(width, height int, input []polytopia.TileData) ([]CanonicalTile, error) {
	expected := width * height
	if len(input) != expected {
		return nil, fmt.Errorf("expected %d tiles but parsed %d", expected, len(input))
	}
	tiles := make([]CanonicalTile, 0, len(input))
	seen := make(map[[2]int]struct{}, len(input))
	for _, tile := range input {
		x, y := tile.WorldCoordinates[0], tile.WorldCoordinates[1]
		if x < 0 || x >= width || y < 0 || y >= height {
			return nil, fmt.Errorf("out-of-bounds coordinate (%d,%d) for %dx%d map", x, y, width, height)
		}
		key := [2]int{x, y}
		if _, exists := seen[key]; exists {
			return nil, fmt.Errorf("duplicate coordinate (%d,%d)", x, y)
		}
		seen[key] = struct{}{}
		canonical := CanonicalTile{
			X: x, Y: y, TerrainID: tile.Terrain, ClimateID: tile.Climate,
			Altitude: tile.Altitude, OwnerID: tile.Owner, Capital: tile.Capital != 0,
			CapitalX: tile.CapitalCoordinates[0], CapitalY: tile.CapitalCoordinates[1],
			HasRoad: tile.HasRoad, HasWaterRoute: tile.HasWaterRoute, TileSkin: tile.TileSkin,
			Flooded: tile.FloodedFlag != 0, FloodedValue: tile.FloodedValue,
		}
		if tile.ResourceExists {
			canonical.Resource = &CanonicalResource{ID: tile.ResourceType}
		}
		if tile.ImprovementExists {
			if tile.ImprovementData == nil {
				return nil, fmt.Errorf("tile (%d,%d) has improvement ID %d but no improvement data", x, y, tile.ImprovementType)
			}
			d := tile.ImprovementData
			canonical.Improvement = &CanonicalImprovement{
				ID: tile.ImprovementType, Level: d.Level, FoundedTurn: d.FoundedTurn,
				CurrentPopulation: d.CurrentPopulation, TotalPopulation: d.TotalPopulation,
				Production: d.Production, BaseScore: d.BaseScore, BorderSize: d.BorderSize,
				UpgradeCount: d.UpgradeCount, ConnectedPlayerCapital: d.ConnectedPlayerCapital,
				CityName: d.CityName, FoundedTribeID: d.FoundedTribe,
				CityRewardIDs: nonNilInts(d.CityRewards), RebellionFlag: d.RebellionFlag,
				RebellionBuffer: nonNilInts(d.RebellionBuffer),
			}
		}
		tiles = append(tiles, canonical)
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if _, ok := seen[[2]int{x, y}]; !ok {
				return nil, fmt.Errorf("missing coordinate (%d,%d)", x, y)
			}
		}
	}
	sort.Slice(tiles, func(i, j int) bool {
		if tiles[i].Y != tiles[j].Y {
			return tiles[i].Y < tiles[j].Y
		}
		return tiles[i].X < tiles[j].X
	})
	return tiles, nil
}

// mapHashInput intentionally excludes source identity, seed, map name, timestamps,
// current game state, and action history. It fingerprints generated map content only.
type mapHashInput struct {
	SchemaVersion int               `json:"schema_version"`
	Width         int               `json:"width"`
	Height        int               `json:"height"`
	Players       []CanonicalPlayer `json:"players"`
	Tiles         []CanonicalTile   `json:"tiles"`
}

func calculateMapSHA256(m *CanonicalMap) (string, error) {
	payload, err := json.Marshal(mapHashInput{schemaVersion, m.Game.MapWidth, m.Game.MapHeight, m.Players, m.Tiles})
	if err != nil {
		return "", fmt.Errorf("serialize canonical map hash input: %w", err)
	}
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:]), nil
}

func nonNilInts(input []int) []int {
	if len(input) == 0 {
		return []int{}
	}
	return append([]int(nil), input...)
}
