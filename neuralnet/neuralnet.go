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
	LossFn       func(y, yPred float64) float64
	LearningRate float64
	layers       []layer
}

func NewNeuralNet(lr float64, lossFn func(y, yPred float64) float64) *NeuralNet {
	return &NeuralNet{
		LearningRate: lr,
		LossFn:       lossFn,
	}
}

func (nn *NeuralNet) Train(x, y []float64) error {
	lenLayers := len(nn.layers)

	if !(lenLayers >= 3 && nn.layers[0].IsInputLayer && nn.layers[lenLayers-1].IsOutputLayer) {
		return errors.New("training requires 1 input layer, n hidden layers, and 1 output layer")
	}

	inputLayer := nn.layers[0]

	if len(x) != inputLayer.Units {
		return fmt.Errorf("train expected %d x length, got %d", inputLayer.Units, len(x))
	}

	outputLayer := nn.layers[lenLayers-1]

	if len(y) != outputLayer.Units {
		return fmt.Errorf("train expected %d y length, got %d", outputLayer.Units, len(y))
	}

	err := nn.forwardPropagate(x, y)

	if err != nil {
		return fmt.Errorf("failed to train model: %w", err)
	}

	err = nn.backwardPropagate(y)

	if err != nil {
		return fmt.Errorf("failed to train model: %w", err)
	}

	return nil
}

func (nn *NeuralNet) forwardPropagate(x, y []float64) error {
	copy(nn.layers[0].Values, x)

	for i := 0; i < len(nn.layers)-1; i++ {
		current := nn.layers[i]
		next := nn.layers[i+1]

		for unit := range next.Units {
			w, err := current.Weights.SliceRow(unit)

			if err != nil {
				return fmt.Errorf("failed to get weights when slicing row: %w", err)
			}

			z, err := vector.Multiply(current.Values, w)

			if err != nil {
				return fmt.Errorf("failed to forward propagate: %w", err)
			}

			z += next.Biases[unit]

			z = next.Activation.Fn(z)

			next.Values[unit] = z
		}
	}

	return nil
}

func (nn *NeuralNet) backwardPropagate(y []float64) error {
	lenLayers := len(nn.layers)

	// Start from output layer and move backward
	for i := lenLayers - 1; i > 0; i-- {
		current := nn.layers[i]
		prev := nn.layers[i-1]

		for i := range current.Units {
			var loss float64

			if current.IsOutputLayer {
				// Output layer error: (y_pred - y_true)
				loss = nn.LossFn(y[i], current.Values[i])
			} else {
				// Hidden layer error: sum of weighted errors from next layer
				for j := range current.NextLayer.Units {
					w, _ := current.Weights.At(j, i)
					loss += current.NextLayer.Gradients[j] * w
				}
			}

			// Apply derivative of activation
			grad := loss * current.Activation.FnPrime(current.Values[i])
			current.Gradients[i] = grad

			// Update weights and biases
			for j := range prev.Units {
				oldWeight, _ := prev.Weights.At(i, j)
				newWeight := oldWeight - nn.LearningRate*grad*prev.Values[j]
				prev.Weights.Set(i, j, newWeight)
			}

			current.Biases[i] -= nn.LearningRate * grad
		}
	}

	return nil
}

func (nn *NeuralNet) AddInputLayer(units int) error {
	if len(nn.layers) > 0 {
		return errors.New("only first layer must be an input layer")
	}

	return nn.addLayer(units, activation.Identity(), true, false)
}

func (nn *NeuralNet) AddHiddenLayer(units int, activation *activation.Activation) error {
	if len(nn.layers) == 0 || !nn.layers[0].IsInputLayer {
		return errors.New("first layer must be an input layer")
	}

	if nn.layers[len(nn.layers)-1].IsOutputLayer {
		return errors.New("cannot add hidden layers after output layer")
	}

	err := nn.addLayer(units, activation, false, false)

	if err != nil {
		return fmt.Errorf("failed to add hidden layer: %w", err)
	}

	return nil
}

func (nn *NeuralNet) AddOutputLayer(units int, activation *activation.Activation) error {
	if len(nn.layers) == 0 || !nn.layers[0].IsInputLayer {
		return errors.New("first layer must be an input layer")
	}

	if nn.layers[len(nn.layers)-1].IsOutputLayer {
		return errors.New("output layer already exists")
	}

	return nn.addLayer(units, activation, false, true)
}

func (nn *NeuralNet) addLayer(units int, activation *activation.Activation, isInputLayer, isOutputLayer bool) error {
	layer, err := newLayer(units, activation, isInputLayer, isOutputLayer)

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

	randomWeights(weights)

	current.Weights = weights

	return nil
}

func randomWeights(weights *matrix.Matrix) {
	for row := range weights.Rows {
		for col := range weights.Cols {
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
