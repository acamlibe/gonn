package neuralnet

import (
	"errors"
	"fmt"
	"gonn/matrix"
	"gonn/neuralnet/activation"
	"gonn/vector"
	"math"
	"math/rand/v2"
)

type NeuralNet struct {
	LossFn       func(y, yPred float64) float64
	LearningRate float64
	layers       []*layer
	hasTrained   bool
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

	err := nn.forwardPropagate(x)

	if err != nil {
		return fmt.Errorf("failed to train model: %w", err)
	}

	err = nn.backwardPropagate(y)

	if err != nil {
		return fmt.Errorf("failed to train model: %w", err)
	}

	nn.hasTrained = true

	return nil
}

func (nn *NeuralNet) Predict(x []float64) ([]float64, error) {
	if !nn.hasTrained {
		return nil, errors.New("neuralnet has not been trained yet")
	}

	err := nn.forwardPropagate(x)

	if err != nil {
		return nil, fmt.Errorf("failed to predict model: %w", err)
	}

	lenLayers := len(nn.layers)
	outputLayer := nn.layers[lenLayers-1]

	return outputLayer.Values, nil
}

func (nn *NeuralNet) forwardPropagate(x []float64) error {
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

	// Start from output layer and move backward
	for i := lenLayers - 1; i > 0; i-- {
		current := nn.layers[i]
		prev := nn.layers[i-1]

		for j := range current.Units {
			var delta float64

			if current.IsOutputLayer {
				// Output layer: derivative of loss
				delta = current.Values[j] - y[j]
			} else {
				// Hidden layer: sum of next layer gradients * corresponding weights
				next := nn.layers[i+1]

				// We need to sum over all units in the next layer
				sum := 0.0
				for k := range next.Units {
					// The weight is from current unit j to next unit k
					w, err := current.Weights.At(k, j) // Note: using current's weights, not next's
					if err != nil {
						return fmt.Errorf("failed to access weight during backprop: %w", err)
					}
					sum += next.Gradients[k] * w
				}
				delta = sum
			}

			// Derivative of activation: dA/dZ
			grad := delta * current.Activation.FnPrime(current.ZValues[j])
			current.Gradients[j] = grad

			// Update weights between prev and current layers
			for k := range prev.Units {
				oldWeight, err := prev.Weights.At(j, k)
				if err != nil {
					return fmt.Errorf("failed to access previous weight: %w", err)
				}

				// Weight update: w = w - learning_rate * gradient * input
				newWeight := oldWeight - nn.LearningRate*grad*prev.Values[k]
				err = prev.Weights.Set(j, k, newWeight)
				if err != nil {
					return fmt.Errorf("failed to set new weight: %w", err)
				}
			}

			// Update bias
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

	nn.layers = append(nn.layers, layer)

	lenLayers := len(nn.layers)

	if lenLayers >= 2 {
		current := nn.layers[lenLayers-2]
		next := nn.layers[lenLayers-1]

		err := connectLayers(current, next)
		if err != nil {
			return fmt.Errorf("failed to connect layers: %w", err)
		}
	}

	return nil
}

func connectLayers(current, next *layer) error {
	current.NextLayer = next

	weights, err := matrix.NewMatrix(next.Units, current.Units)

	if err != nil {
		return fmt.Errorf("failed to initialize weights: %w", err)
	}

	err = randomWeights(weights)
	if err != nil {
		return fmt.Errorf("failed to initialize weights: %w", err)
	}

	current.Weights = weights

	return nil
}

func randomWeights(weights *matrix.Matrix) error {
	for row := range weights.Rows {
		for col := range weights.Cols {
			err := weights.Set(row, col, glorotInit(weights.Rows, weights.Cols))
			if err != nil {
				return fmt.Errorf("failed to set matrix at (%d, %d): %w", row, col, err)
			}
		}
	}

	return nil
}

func randomBiases(units int) []float64 {
	if units < 1 {
		panic("failed to get random vector, invalid units length")
	}

	vec := make([]float64, units)

	for i := range vec {
		vec[i] = rand.Float64()
	}

	return vec
}

func glorotInit(rows, cols int) float64 {
	limit := math.Sqrt(6.0 / float64(rows+cols))
	return rand.Float64()*2*limit - limit
}
