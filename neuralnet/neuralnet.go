package neuralnet

import (
	"errors"
	"fmt"
	"gonn/matrix"
	"gonn/neuralnet/activation"
	"gonn/vector"
	"math/rand/v2"
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

	return nn.forwardPropagate(x, y)
}

func (nn *NeuralNet) forwardPropagate(x, y []float64) error {
	copy(nn.layers[0].Values, x)

	for i := 0; i < len(nn.layers) - 1; i++ {
		current := nn.layers[i]
		next := nn.layers[i+1]

		for unit := range(next.Units) {
			weights, err := current.Weights.SliceRow(unit)

			if err != nil {
				return fmt.Errorf("failed to get weights when slicing row: %w", err)
			}

			newVal, err := vector.Multiply(current.Values, weights)

			if err != nil {
				return fmt.Errorf("failed to forward propagate: %w", err)
			}

			newVal += next.Biases[unit]

			newVal = next.ActivationFn(newVal)

			next.Values[unit] = newVal;
		}
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

	if !isInputLayer {
		layer.Biases = randomBiases(units)
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
	for row := range(weights.Rows) {
		for col := range(weights.Cols) {
			weights.Set(row, col, randomValue())
		}
	}
}

func randomBiases(units int) []float64 {
	if units < 1 {
		panic("failed to get random vector, invalid units length")
	}

	vec := make([]float64, units)

	for i := range vec {
		vec[i] = randomValue()
	}

	return vec
}

func randomValue() float64 {
	return float64(rand.IntN(8) + 1)
}
