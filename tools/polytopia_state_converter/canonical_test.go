package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	polytopia "github.com/samuelyuan/polytopiamapmodelgo"
)

func fixtureSave() *polytopia.PolytopiaSaveOutput {
	city := &polytopia.ImprovementData{
		Level: 1, FoundedTurn: 0, CurrentPopulation: 0, TotalPopulation: 2,
		Production: 1, BaseScore: 100, BorderSize: 1, UpgradeCount: 0,
		ConnectedPlayerCapital: 1, HasCityName: 1, CityName: "Lakgru",
		FoundedTribe: 4, CityRewards: []int{7}, RebellionFlag: 1, RebellionBuffer: []int{8, 9},
	}
	initial := [][]polytopia.TileData{{
		{WorldCoordinates: [2]int{1, 0}, Terrain: 5, Climate: 4, ResourceExists: true, ResourceType: 999},
		{WorldCoordinates: [2]int{0, 0}, Terrain: 3, Climate: 3, Owner: 1, Capital: 1, CapitalCoordinates: [2]int{0, 0}, ImprovementExists: true, ImprovementType: 1, ImprovementData: city},
	}}
	current := [][]polytopia.TileData{{
		{WorldCoordinates: [2]int{0, 0}, Terrain: 99},
		{WorldCoordinates: [2]int{1, 0}, Terrain: 99},
	}}
	return &polytopia.PolytopiaSaveOutput{
		GameVersion: 122,
		MapHeaderOutput: polytopia.MapHeaderOutput{
			MapHeaderInput: polytopia.MapHeaderInput{Version1: 122, Seed: 42},
			MapName:        "Fixture", MapWidth: 2, MapHeight: 1, MapSquareSize: 2,
		},
		InitialTileData:   initial,
		InitialPlayerData: []polytopia.PlayerData{{PlayerId: 1, Tribe: 4, StartTileCoordinates: [2]int{0, 0}, Currency: 5, NumCities: 1, AvailableTech: []int{3, 1}}},
		TileData:          current,
		PlayerData:        []polytopia.PlayerData{{PlayerId: 1, Tribe: 99, Currency: 999}},
	}
}

func TestCanonicalizeUsesInitialStateAndOrdersTiles(t *testing.T) {
	m, err := canonicalizeSave(fixtureSave(), SourceIdentity{Filename: "fixture.state"})
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Tiles) != 2 || m.Game.MapWidth != 2 || m.Game.MapHeight != 1 {
		t.Fatalf("unexpected dimensions/tile count: %#v", m.Game)
	}
	if m.Tiles[0].X != 0 || m.Tiles[1].X != 1 {
		t.Fatalf("tiles are not row-major: %#v", m.Tiles)
	}
	if m.Tiles[0].TerrainID != 3 || m.Players[0].TribeID != 4 || m.Players[0].Currency != 5 {
		t.Fatal("current TileData/PlayerData leaked into canonical output")
	}
	if m.Tiles[0].Improvement == nil || m.Tiles[0].Improvement.CityName != "Lakgru" || m.Tiles[0].Improvement.CityRewardIDs[0] != 7 {
		t.Fatalf("improvement data lost: %#v", m.Tiles[0].Improvement)
	}
	if m.Tiles[0].Resource != nil || m.Tiles[1].Resource == nil || m.Tiles[1].Resource.ID != 999 {
		t.Fatal("resource presence/unknown ID was not preserved")
	}
	if got := m.Players[0].AvailableTechnologyIDs; got[0] != 1 || got[1] != 3 {
		t.Fatalf("technologies not canonicalized: %v", got)
	}
}

func TestCoordinateValidation(t *testing.T) {
	duplicate := []polytopia.TileData{{WorldCoordinates: [2]int{0, 0}}, {WorldCoordinates: [2]int{0, 0}}}
	if _, err := canonicalizeTiles(2, 1, duplicate); err == nil || !strings.Contains(err.Error(), "duplicate coordinate") {
		t.Fatalf("expected duplicate error, got %v", err)
	}
	if _, err := canonicalizeInitialTileRows(2, 1, [][]polytopia.TileData{{{WorldCoordinates: [2]int{0, 0}}}}); err == nil || !strings.Contains(err.Error(), "expected 2 tiles") {
		t.Fatalf("expected missing tile error, got %v", err)
	}
	outOfBounds := []polytopia.TileData{{WorldCoordinates: [2]int{0, 0}}, {WorldCoordinates: [2]int{2, 0}}}
	if _, err := canonicalizeTiles(2, 1, outOfBounds); err == nil || !strings.Contains(err.Error(), "out-of-bounds") {
		t.Fatalf("expected bounds error, got %v", err)
	}
}

func TestMapHashAndJSONDeterminism(t *testing.T) {
	a, err := canonicalizeSave(fixtureSave(), SourceIdentity{Filename: "a.state", SHA256: "a"})
	if err != nil {
		t.Fatal(err)
	}
	bSave := fixtureSave()
	bSave.MapHeaderOutput.MapHeaderInput.Seed = 999
	bSave.MapHeaderOutput.MapName = "Different"
	b, err := canonicalizeSave(bSave, SourceIdentity{Filename: "b.state", SHA256: "b"})
	if err != nil {
		t.Fatal(err)
	}
	if a.MapSHA256 != b.MapSHA256 {
		t.Fatal("source identity or non-map metadata affected map hash")
	}
	b.Tiles[0].TerrainID++
	changed, err := calculateMapSHA256(b)
	if err != nil {
		t.Fatal(err)
	}
	if changed == a.MapSHA256 {
		t.Fatal("terrain change did not affect map hash")
	}
	b.Tiles[0] = a.Tiles[0]
	b.Tiles[1].Resource.ID++
	changed, _ = calculateMapSHA256(b)
	if changed == a.MapSHA256 {
		t.Fatal("resource change did not affect map hash")
	}
	b.Tiles[1] = a.Tiles[1]
	b.Tiles[0].Improvement.Level++
	changed, _ = calculateMapSHA256(b)
	if changed == a.MapSHA256 {
		t.Fatal("improvement change did not affect map hash")
	}
	one, _ := marshalCanonical(a, true)
	two, _ := marshalCanonical(a, true)
	if string(one) != string(two) {
		t.Fatal("JSON bytes are not deterministic")
	}
	var decoded CanonicalMap
	if err := json.Unmarshal(one, &decoded); err != nil {
		t.Fatal(err)
	}
}

func TestSourceSHA256AndRealParserIntegration(t *testing.T) {
	path := filepath.Clean(filepath.Join("..", "..", "data", "polytopia_maps", "raw_states", "map_000001.state"))
	contents, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		t.Skip("local harvested fixture is not available")
	}
	if err != nil {
		t.Fatal(err)
	}
	m, err := convertState(path)
	if err != nil {
		t.Fatal(err)
	}
	expected := sha256.Sum256(contents)
	if m.Source.SHA256 != hex.EncodeToString(expected[:]) || m.Source.SizeBytes != int64(len(contents)) {
		t.Fatal("source identity mismatch")
	}
	if m.Game.GameVersion != 122 || m.Game.MapWidth != 11 || m.Game.MapHeight != 11 || len(m.Tiles) != 121 {
		t.Fatalf("unexpected real fixture metadata: %#v tiles=%d", m.Game, len(m.Tiles))
	}
}
