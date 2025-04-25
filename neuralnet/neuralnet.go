package neuralnet

import (
	"errors"
	"fmt"
	"gonn/matrix"
	"gonn/neuralnet/activation"
	"gonn/vector"
)

type NeuralNet struct {
	layers []layer
}

func NewNeuralNet() (*NeuralNet) {
	return &NeuralNet{}
}

func (nn *NeuralNet) Train(x, y []float64) error {
	lenLayers := len(nn.layers)

	if !(lenLayers >= 3 && nn.layers[0].IsInputLayer && nn.layers[lenLayers-1].IsOutputLayer) {
		return errors.New("training requires 1 input layer, n hidden layers, and 1 output layer")
	}

	inputLayer := nn.layers[0]
	//hiddenLayers := nn.layers[1:lenLayers-1]
	outputLayer := nn.layers[lenLayers-1]

	if len(x) != inputLayer.Units {
		return fmt.Errorf("train expected %d x length, got %d", inputLayer.Units, len(x))
	}

	if (len(y) != outputLayer.Units) {
		return fmt.Errorf("train expected %d y length, got %d", outputLayer.Units, len(y))
	}

	return nn.forward(x, y)
}

func (nn *NeuralNet) forward(x, y []float64) error {
	copy(nn.layers[0].Values, x)

	for i := 0; i < len(nn.layers) - 1; i++ {
		current := nn.layers[i]
		next := nn.layers[i+1]

		next.Values = vector.Multiply()
	}

	return nil
}

func (nn *NeuralNet) AddInputLayer(units int) error {
	if len(nn.layers) > 0 {
		return errors.New("only first layer must be an input layer")
	}

	return nn.addLayer(units, activation.None, true, false)
}

func (nn *NeuralNet) AddHiddenLayer(units int, activationFn func(float64) float64) error {
	if len(nn.layers) == 0 || !nn.layers[0].IsInputLayer {
		return errors.New("first layer must be an input layer")
	}

	if nn.layers[len(nn.layers)-1].IsOutputLayer {
		return errors.New("cannot add hidden layers after output layer")
	}

	err := nn.addLayer(units, activationFn, false, false)

	if err != nil {
		return fmt.Errorf("failed to add hidden layer: %w", err)
	}

	return nil
}

func (nn *NeuralNet) AddOutputLayer(units int, activationFn func(float64) float64) error {
	if len(nn.layers) == 0 || !nn.layers[0].IsInputLayer {
		return errors.New("first layer must be an input layer")
	}

	if nn.layers[len(nn.layers)-1].IsOutputLayer {
		return errors.New("output layer already exists")
	}

	return nn.addLayer(units, activationFn, false, true)
}

func (nn *NeuralNet) addLayer(units int, activationFn func(float64) float64, isInputLayer, isOutputLayer bool) error {
	layer, err := newLayer(units, activationFn, isInputLayer, isOutputLayer)

	if err != nil {
		return fmt.Errorf("failed to add input layer: %w", err)
	}

	nn.layers = append(nn.layers, *layer)

	lenLayers := len(nn.layers)

	if lenLayers >= 2 {
		current := nn.layers[lenLayers-2]
		next := nn.layers[lenLayers-1]

		connectLayers(&current, &next)
	}

	return nil
}

func connectLayers(current, next *layer) error {
	current.NextLayer = next

	weights, err := matrix.NewMatrix(next.Units, current.Units)

	if err != nil {
		return fmt.Errorf("failed to initialize weights: %w", err)
	}

	current.Weights = weights

	return nil
}

func randomWeights(weights *matrix.Matrix) {

}
