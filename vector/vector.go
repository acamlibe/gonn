package vector

import "fmt"

func Multiply(v1, v2 []float64) (float64, error) {
	nV := len(v1)
	nV2 := len(v2)

	if nV != nV2 {
		return 0, fmt.Errorf("vectors must be the same size to multiply - expected length %d, got %d",
			nV, nV2)
	}

	sum := float64(0)

	for i := range v1 {
		sum += v1[i] * v2[i]
	}

	return sum, nil
}

func ApplyFn(v []float64, fn func(float64) float64) {
	for i, s := range v {
		v[i] = fn(s)
	}
}
