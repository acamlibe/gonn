package activation

import "math"

type Activation struct {
	Fn      func(z float64) float64
	FnPrime func(z float64) float64
}

func IdentityRound() *Activation {
	return &Activation{
		Fn:      identityRound,
		FnPrime: identityPrime,
	}
}


func Identity() *Activation {
	return &Activation{
		Fn:      identity,
		FnPrime: identityPrime,
	}
}

func identity(z float64) float64 {
	return z
}

func identityRound(z float64) float64 {
	return math.Round(z)
}

func identityPrime(_ float64) float64 {
	return 1
}

func ReLU() *Activation {
	return &Activation{
		Fn:      relu,
		FnPrime: reluPrime,
	}
}

func relu(z float64) float64 {
	return math.Max(0, z)
}

func reluPrime(z float64) float64 {
	if z > 0 {
		return 1
	}
	return 0
}

func Sigmoid() *Activation {
	return &Activation{
		Fn:      sigmoid,
		FnPrime: sigmoidPrime,
	}
}

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func sigmoidPrime(z float64) float64 {
	sig := sigmoid(z)
	return sig * (1 - sig)
}
