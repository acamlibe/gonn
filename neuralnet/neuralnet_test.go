package neuralnet

import (
	"errors"
	"gonn/neuralnet/activation"
	"gonn/neuralnet/loss"
	"gonn/reader/csv"
	"gonn/sample"
	"math"
	"strconv"
	"strings"
	"testing"
)

func simpleParse(s *string) (float64, error) {
	v, err := strconv.ParseFloat(*s, 64)

	if err != nil {
		return 0, err
	}

	return v, nil
}

func classParse(s *string) (float64, error) {
	switch {
	case strings.EqualFold(*s, "Setosa"):
		return float64(0), nil
	case strings.EqualFold(*s, "Versicolor"):
		return float64(1), nil
	case strings.EqualFold(*s, "Virginica"):
		return float64(2), nil
	default:
		return float64(0), errors.New("undefined iris flower class")
	}
}

func TestRegression(t *testing.T) {
	path, err := sample.GetSampleFilePath("winequality-red.csv")

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	r := csv.NewReader(path, true, ';')

	r.DefineColumn(0, "fixed acidity", simpleParse)
	r.DefineColumn(1, "volatile acidity", simpleParse)
	r.DefineColumn(2, "citric acid", simpleParse)
	r.DefineColumn(3, "residual sugar", simpleParse)
	r.DefineColumn(4, "chlorides", simpleParse)
	r.DefineColumn(5, "free sulfur dioxide", simpleParse)
	r.DefineColumn(6, "total sulfur dioxide", simpleParse)
	r.DefineColumn(7, "density", simpleParse)
	r.DefineColumn(8, "pH", simpleParse)
	r.DefineColumn(9, "sulphates", simpleParse)
	r.DefineColumn(10, "alcohol", simpleParse)
	r.DefineColumn(11, "quality", simpleParse)

	err = r.ReadTable()

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	m := r.DataTable.Matrix

	lr := 0.00001

	nn := NewNeuralNet(lr, loss.MSE)

	err = nn.AddInputLayer(11)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	err = nn.AddHiddenLayer(11, activation.ReLU())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	err = nn.AddOutputLayer(1, activation.IdentityRound())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	trainRows := int(math.Floor(float64(m.Rows) * 0.95))
	testRows := m.Rows - trainRows

	for row := range trainRows {
		v, err := m.SliceRow(row)

		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		x := v[0:11]
		y := v[11:12]

		err = nn.Train(x, y)

		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
	}

	sumMapes := float64(0)

	for row := trainRows; row < trainRows+testRows; row++ {
		v, err := m.SliceRow(row)

		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		x := v[0:11]
		y := v[11:12]

		yPred, err := nn.Predict(x)

		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		// With this safer MAPE calculation:
		var mape float64
		if y[0] != 0 {
		    mape = math.Abs((y[0] - yPred[0]) / y[0])
		} else {
		    // Handle case where y[0] is 0
		    // You might want to use absolute error instead or skip this sample
		    mape = math.Abs(y[0] - yPred[0])
		    // Or: continue to skip this sample
		}

		t.Logf("MAPE %f, y %f, ypred %f", mape, y[0], yPred[0])

		sumMapes += mape
	}

	t.Logf("Overall MAPE: %f", sumMapes/float64(testRows))
}
