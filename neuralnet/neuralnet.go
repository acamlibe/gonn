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

	for i := range len(nn.layers) - 1 {
		current := nn.layers[i]
		next := nn.layers[i+1]

		for unit := range next.Units {
			w, err := current.Weights.SliceRow(unit)

			if err != nil {
				return fmt.Errorf("failed to get weights when slicing row: %w", err)
			}

			dot, err := vector.Multiply(current.Values, w)

			if err != nil {
				return fmt.Errorf("failed to forward propagate: %w", err)
			}

			z := dot + next.Biases[unit]

			next.ZValues[unit] = z
			next.Values[unit] = next.Activation.Fn(z)
		}
	}

	return nil
}

func (nn *NeuralNet) backwardPropagate(y []float64) error {
	lenLayers := len(nn.layers)

	for i := lenLayers - 1; i > 0; i-- {
		current := nn.layers[i]
		prev := nn.layers[i-1]

		for j := range current.Units {
			var delta float64

			if current.IsOutputLayer {
				// Derivative of loss: dL/dy_pred
				delta = current.Values[j] - y[j]
			} else {
				// Hidden layer: sum of next layer gradients * corresponding weights
				next := current.NextLayer
				for k := range next.Units {
					w, err := next.Weights.At(k, j)
					if err != nil {
						return fmt.Errorf("failed to access weight during backprop: %w", err)
					}
					delta += next.Gradients[k] * w
				}
			}

			// Derivative of activation: dA/dZ
			grad := delta * current.Activation.FnPrime(current.ZValues[j])
			current.Gradients[j] = grad

			// Update weights and biases
			for k := range prev.Units {
				oldWeight, err := prev.Weights.At(j, k)
				if err != nil {
					return fmt.Errorf("failed to access previous weight: %w", err)
				}

				newWeight := oldWeight - nn.LearningRate*grad*prev.Values[k]
				err = prev.Weights.Set(j, k, newWeight)
				if err != nil {
					return fmt.Errorf("failed to set new weight: %w", err)
				}
			}

			current.Biases[j] -= nn.LearningRate * grad
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
	return rand.Float64()*2 - 1
}
