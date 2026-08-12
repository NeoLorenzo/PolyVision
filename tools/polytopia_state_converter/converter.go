package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	polytopia "github.com/samuelyuan/polytopiamapmodelgo"
)

func convertState(inputPath string) (*CanonicalMap, error) {
	contents, err := os.ReadFile(inputPath)
	if err != nil {
		return nil, fmt.Errorf("read source: %w", err)
	}
	if len(contents) == 0 {
		return nil, fmt.Errorf("source file is empty")
	}
	sum := sha256.Sum256(contents)
	source := SourceIdentity{
		Filename:  filepath.Base(inputPath),
		SHA256:    hex.EncodeToString(sum[:]),
		SizeBytes: int64(len(contents)),
	}
	save, err := polytopia.ReadPolytopiaCompressedInitialFile(inputPath)
	if err != nil {
		return nil, fmt.Errorf("parse compressed state: %w", err)
	}
	return canonicalizeSave(save, source)
}

func marshalCanonical(m *CanonicalMap, pretty bool) ([]byte, error) {
	var data []byte
	var err error
	if pretty {
		data, err = json.MarshalIndent(m, "", "  ")
	} else {
		data, err = json.Marshal(m)
	}
	if err != nil {
		return nil, fmt.Errorf("serialize canonical JSON: %w", err)
	}
	return append(data, '\n'), nil
}

func writeAtomically(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".polytopia-state-*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary output: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err = bytes.NewReader(data).WriteTo(tmp); err != nil {
		tmp.Close()
		return fmt.Errorf("write temporary output: %w", err)
	}
	if err = tmp.Sync(); err != nil {
		tmp.Close()
		return fmt.Errorf("sync temporary output: %w", err)
	}
	if err = tmp.Close(); err != nil {
		return fmt.Errorf("close temporary output: %w", err)
	}
	if err = os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("replace output: %w", err)
	}
	return nil
}
