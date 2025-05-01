package loss

import "math"

func MSE(y, yPred float64) float64 {
	return math.Pow(y - yPred, 2)
}
