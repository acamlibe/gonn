package neuralnet

import (
	"fmt"
	"gonn/matrix"
	"gonn/neuralnet/activation"
)

type layer struct {
	Units         int
	Values        []float64
	ZValues       []float64
	Weights       *matrix.Matrix
	Biases        []float64
	Gradients     []float64
	NextLayer     *layer
	Activation    *activation.Activation
	IsInputLayer  bool
	IsOutputLayer bool
}

func newLayer(units int, activation *activation.Activation, isInputLayer bool, isOutputLayer bool) (*layer, error) {
	if units < 1 {
		return nil, fmt.Errorf("layer units must be greater than 0, got %d", units)
	}

	layer := layer{
		Units:         units,
		Values:        make([]float64, units),
		ZValues:       make([]float64, units),
		Gradients:     make([]float64, units),
		Activation:    activation,
		IsInputLayer:  isInputLayer,
		IsOutputLayer: isOutputLayer,
	}

	return &layer, nil
}
