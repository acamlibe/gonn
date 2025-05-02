package sample

import (
	"fmt"
	"os"
	"path/filepath"
)

func GetSampleFilePath(fileName string) (string, error) {
	cwd, err := os.Getwd()

	if err != nil {
		return "", fmt.Errorf("failed to get working directory: %w", err)
	}

	// Walk up to project root if needed
	for filepath.Base(cwd) != "gonn" && len(cwd) > 1 {
		cwd = filepath.Dir(cwd)
	}

	return filepath.Join(cwd, "sample", fileName), nil
}
