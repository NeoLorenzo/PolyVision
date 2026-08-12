package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBatchNamingSkipAndOverwrite(t *testing.T) {
	input, output := t.TempDir(), t.TempDir()
	for _, name := range []string{"map_000002.state", "map_000001.state", "ignore.txt"} {
		if err := os.WriteFile(filepath.Join(input, name), []byte(name), 0644); err != nil {
			t.Fatal(err)
		}
	}
	original := invokeWorker
	t.Cleanup(func() { invokeWorker = original })
	var calls []string
	invokeWorker = func(input, output string, pretty, verbose bool) error {
		calls = append(calls, filepath.Base(input)+":"+filepath.Base(output))
		m, err := canonicalizeSave(fixtureSave(), SourceIdentity{Filename: filepath.Base(input), SHA256: filepath.Base(input)})
		if err != nil {
			return err
		}
		data, err := marshalCanonical(m, pretty)
		if err != nil {
			return err
		}
		return writeAtomically(output, data)
	}
	opt := options{input: input, output: output, pretty: true}
	if code := runBatch(opt); code != 0 {
		t.Fatalf("batch exited %d", code)
	}
	if len(calls) != 2 || calls[0] != "map_000001.state:map_000001.json" || calls[1] != "map_000002.state:map_000002.json" {
		t.Fatalf("unexpected deterministic calls: %v", calls)
	}
	if code := runBatch(opt); code != 0 || len(calls) != 2 {
		t.Fatalf("existing outputs were not skipped: code=%d calls=%v", code, calls)
	}
	opt.overwrite = true
	if code := runBatch(opt); code != 0 || len(calls) != 4 {
		t.Fatalf("overwrite did not reconvert: code=%d calls=%v", code, calls)
	}
}

func TestSingleExistingOutputRequiresOverwrite(t *testing.T) {
	temp := t.TempDir()
	input := filepath.Join(temp, "map.state")
	output := filepath.Join(temp, "map.json")
	os.WriteFile(input, []byte("state"), 0644)
	os.WriteFile(output, []byte("existing"), 0644)
	original := invokeWorker
	t.Cleanup(func() { invokeWorker = original })
	called := false
	invokeWorker = func(_, _ string, _, _ bool) error { called = true; return nil }
	if code := runSingle(options{input: input, output: output}); code == 0 || called {
		t.Fatal("existing output was overwritten without --overwrite")
	}
	if code := runSingle(options{input: input, output: output, overwrite: true}); code != 0 || !called {
		t.Fatal("--overwrite did not invoke conversion")
	}
}
