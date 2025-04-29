package activation

import "math"

func None(x float64) float64 {
	return x
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
