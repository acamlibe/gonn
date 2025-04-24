package matrix

import "fmt"

type Vector []float64

func (v *Vector) Multiply(v2 *Vector) (float64, error) {
	nV := len(*v)
	nV2 := len(*v2)

	if nV != nV2 {
		return 0, fmt.Errorf("vectors must be the same size to multiply - expected length %d, got %d",
			nV, nV2)
	}

	sum := float64(0)

	for i := range *v {
		sum += (*v)[i] * (*v2)[i]
	}

	return sum, nil
}
