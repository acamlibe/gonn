package neuralnet

import (
	"fmt"
	"math/rand/v2"
)

type Layer struct {
	Units         int
	Values        []float64
	Weights       []float64
	Biases        []float64
	nextLayer     *Layer
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
		Values:        make([]float64, units),
		Weights:       randomVector(units),
		Biases:        randomVector(units),
		isInputLayer:  isInputLayer,
		isOutputLayer: isOutputLayer,
	}

	return &layer, nil
}

func randomVector(units int) []float64 {
	if units < 1 {
		panic("failed to get random vector, invalid units length")
	}

	vec := make([]float64, units)

	for i := range vec {
		vec[i] = float64(rand.IntN(8) + 1)
	}

	return vec
}
