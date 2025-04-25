package neuralnet

import (
	"fmt"
	"gonn/matrix"
	"math/rand/v2"
)

type Layer struct {
	Units         int
	Values        matrix.Vector
	Weights       *matrix.Matrix
	Biases        matrix.Vector
	NextLayer     *Layer
	isInputLayer  bool
	isOutputLayer bool
}

func NewInputLayer(units int) (*Layer, error) {
	return newLayer(units, true, false)
}

func NewHiddenLayer(units int) (*Layer, error) {
	return newLayer(units, false, false)
}

func NewOutputLayer(units int) (*Layer, error) {
	return newLayer(units, false, true)
}

func newLayer(units int, isInputLayer bool, isOutputLayer bool) (*Layer, error) {
	if units < 1 {
		return nil, fmt.Errorf("layer units must be greater than 0, got %d", units)
	}

	layer := Layer{
		Units:         units,
		Values:        make(matrix.Vector, units),
		Biases:        randomVector(units),
		isInputLayer:  isInputLayer,
		isOutputLayer: isOutputLayer,
	}

	return &layer, nil
}

func randomVector(units int) matrix.Vector {
	if units < 1 {
		panic("failed to get random vector, invalid units length")
	}

	vec := make(matrix.Vector, units)

	for i := range vec {
		vec[i] = float64(rand.IntN(8) + 1)
	}

	return vec
}
