package loss

func MSE(y, yPred float64) float64 {
	return (y - yPred) * (y - yPred)
}
