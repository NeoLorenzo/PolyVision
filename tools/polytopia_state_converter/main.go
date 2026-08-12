package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

type options struct {
	input, output                      string
	overwrite, pretty, verbose, worker bool
}
type counters struct {
	found, converted, skipped, failed int
	rawHashes, mapHashes              map[string]struct{}
	versions, dimensions              map[string]int
}

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	fs := flag.NewFlagSet("polytopia_state_converter", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var opt options
	fs.StringVar(&opt.input, "input", "", "input .state file or directory")
	fs.StringVar(&opt.output, "output", "", "output .json file or directory")
	fs.BoolVar(&opt.overwrite, "overwrite", false, "replace existing JSON files")
	fs.BoolVar(&opt.pretty, "pretty", true, "pretty-print JSON")
	fs.BoolVar(&opt.verbose, "verbose", false, "print parser worker diagnostics")
	fs.BoolVar(&opt.worker, "worker", false, "internal parser worker")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if opt.input == "" || opt.output == "" {
		fmt.Fprintln(os.Stderr, "error: --input and --output are required")
		return 2
	}
	if opt.worker {
		return runWorker(opt)
	}
	info, err := os.Stat(opt.input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: inspect input: %v\n", err)
		return 1
	}
	if info.IsDir() {
		return runBatch(opt)
	}
	return runSingle(opt)
}

func runWorker(opt options) int {
	m, err := convertState(opt.input)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	data, err := marshalCanonical(m, opt.pretty)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if err := writeAtomically(opt.output, data); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	return 0
}

var invokeWorker = invokeWorkerProcess

func invokeWorkerProcess(input, output string, pretty, verbose bool) error {
	executable, err := os.Executable()
	if err != nil {
		return fmt.Errorf("locate converter executable: %w", err)
	}
	args := []string{"--worker", "--input", input, "--output", output, fmt.Sprintf("--pretty=%t", pretty)}
	cmd := exec.Command(executable, args...)
	combined, err := cmd.CombinedOutput()
	if verbose && len(combined) > 0 {
		fmt.Fprint(os.Stderr, string(combined))
	}
	if err != nil {
		reason := strings.TrimSpace(string(combined))
		if reason == "" {
			reason = err.Error()
		}
		return fmt.Errorf("%s", reason)
	}
	return nil
}

func runSingle(opt options) int {
	if !strings.EqualFold(filepath.Ext(opt.input), ".state") {
		fmt.Fprintln(os.Stderr, "error: input file must have a .state extension")
		return 1
	}
	if info, err := os.Stat(opt.output); err == nil && info.IsDir() {
		opt.output = filepath.Join(opt.output, strings.TrimSuffix(filepath.Base(opt.input), filepath.Ext(opt.input))+".json")
	}
	if !opt.overwrite {
		if _, err := os.Stat(opt.output); err == nil {
			fmt.Fprintf(os.Stderr, "error: output exists (use --overwrite): %s\n", opt.output)
			return 1
		}
	}
	if err := invokeWorker(opt.input, opt.output, opt.pretty, opt.verbose); err != nil {
		fmt.Fprintf(os.Stderr, "error: %s: %v\n", filepath.Base(opt.input), err)
		return 1
	}
	fmt.Printf("%s -> %s ✓\n", filepath.Base(opt.input), filepath.Base(opt.output))
	return 0
}

func runBatch(opt options) int {
	entries, err := os.ReadDir(opt.input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: read input directory: %v\n", err)
		return 1
	}
	var files []string
	for _, entry := range entries {
		if !entry.IsDir() && strings.EqualFold(filepath.Ext(entry.Name()), ".state") {
			files = append(files, entry.Name())
		}
	}
	sort.Strings(files)
	stats := counters{found: len(files), rawHashes: map[string]struct{}{}, mapHashes: map[string]struct{}{}, versions: map[string]int{}, dimensions: map[string]int{}}
	fmt.Println("POLYVISION — POLYTOPIA STATE EXTRACTOR")
	fmt.Printf("\nInput:  %s\nOutput: %s\n\nFound %d .state files.\n\n", opt.input, opt.output, len(files))
	if err := os.MkdirAll(opt.output, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "error: create output directory: %v\n", err)
		return 1
	}
	var failures []string
	for i, name := range files {
		input := filepath.Join(opt.input, name)
		output := filepath.Join(opt.output, strings.TrimSuffix(name, filepath.Ext(name))+".json")
		if !opt.overwrite {
			if _, err := os.Stat(output); err == nil {
				stats.skipped++
				collectOutputStats(output, &stats)
				fmt.Printf("[%d/%d] %s -> %s skipped (exists)\n", i+1, len(files), name, filepath.Base(output))
				continue
			}
		}
		if err := invokeWorker(input, output, opt.pretty, opt.verbose); err != nil {
			stats.failed++
			failures = append(failures, fmt.Sprintf("%s: %v", name, err))
			fmt.Printf("[%d/%d] %s FAILED\n", i+1, len(files), name)
			continue
		}
		stats.converted++
		collectOutputStats(output, &stats)
		fmt.Printf("[%d/%d] %s -> %s ✓\n", i+1, len(files), name, filepath.Base(output))
	}
	fmt.Println("\nComplete.")
	fmt.Printf("\nFound:           %d\nConverted:       %d\nSkipped existing:%d\nFailed:          %d\nUnique raw saves:%d\nUnique maps:     %d\n", stats.found, stats.converted, stats.skipped, stats.failed, len(stats.rawHashes), len(stats.mapHashes))
	printSummary("Game versions", stats.versions)
	printSummary("Dimensions", stats.dimensions)
	if len(failures) > 0 {
		fmt.Fprintln(os.Stderr, "\nFailures:")
		for _, failure := range failures {
			fmt.Fprintln(os.Stderr, "  "+failure)
		}
		return 1
	}
	return 0
}

func collectOutputStats(path string, stats *counters) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var m CanonicalMap
	if jsonUnmarshal(data, &m) != nil {
		return
	}
	stats.rawHashes[m.Source.SHA256] = struct{}{}
	stats.mapHashes[m.MapSHA256] = struct{}{}
	stats.versions[fmt.Sprint(m.Game.GameVersion)]++
	stats.dimensions[fmt.Sprintf("%d×%d", m.Game.MapWidth, m.Game.MapHeight)]++
}

var jsonUnmarshal = func(data []byte, value any) error { return json.Unmarshal(data, value) }

func printSummary(label string, values map[string]int) {
	if len(values) == 0 {
		return
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	fmt.Printf("\n%s:\n", label)
	for _, key := range keys {
		fmt.Printf("  %s: %d\n", key, values[key])
	}
}
