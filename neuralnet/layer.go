package neuralnet

import (
	"fmt"
	"gonn/matrix"
	"math/rand/v2"
)

type layer struct {
	Units         int
	Values        []float64
	Weights       *matrix.Matrix
	Biases        []float64
	NextLayer     *layer
	ActivationFn  func(float64) float64
	IsInputLayer  bool
	IsOutputLayer bool
}

func newLayer(units int, activationFn func(float64) float64, isInputLayer bool, isOutputLayer bool) (*layer, error) {
	if units < 1 {
		return nil, fmt.Errorf("layer units must be greater than 0, got %d", units)
	}

	layer := layer{
		Units:         units,
		Values:        make([]float64, units),
		ActivationFn:  activationFn,
		IsInputLayer:  isInputLayer,
		IsOutputLayer: isOutputLayer,
	}

	return &layer, nil
}
