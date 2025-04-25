package vector

import (
	"testing"
)

func TestMultiply(t *testing.T) {
	v1 := []float64{2, 4, 3}
	v2 := []float64{4, 1, 5}

	v, err := Multiply(v1, v2)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if v != float64(27) {
		t.Errorf("expected vector multiply to return %f scalar value, got %f", float64(27), v)
	}
}

func TestMultiplyError(t *testing.T) {
	v1 := []float64{2, 4, 3}
	v2 := []float64{4, 1, 5, 6}

	_, err := Multiply(v1, v2)

	if err == nil {
		t.Errorf("expected error, got nil")
	}
}
