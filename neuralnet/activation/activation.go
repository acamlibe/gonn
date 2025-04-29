package activation

import "math"

func None(x float64) float64 {
	return x
}

func NonePrime(x float64) float64 {
	return 1
}

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func ReLUPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidPrime(x float64) float64 {
	sig := Sigmoid(x)
	return sig * (1 - sig)
}
