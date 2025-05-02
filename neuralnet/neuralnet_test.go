package neuralnet

import (
	"errors"
	"gonn/neuralnet/activation"
	"gonn/neuralnet/loss"
	"gonn/reader/csv"
	"gonn/sample"
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

	hasHeader := true

	r := csv.NewReader(path, hasHeader, ';')

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

	lr := 0.1

	nn := NewNeuralNet(lr, loss.MSE)

	nn.AddInputLayer(11)
	nn.AddHiddenLayer(20, activation.ReLU())
	nn.AddOutputLayer(1, activation.Identity())

	for row := range m.Rows {
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
}
