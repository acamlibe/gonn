package neuralnet

import (
	"errors"
	"gonn/reader/csv"
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
	//sepal.length, sepal.width, petal.length, petal.width, variety

	path := "../sample/winequality-red.csv"
	hasHeader := true

	r := csv.NewReader(path, hasHeader, ';')

	r.DefineColumn(0, "sepal.length", simpleParse)
	r.DefineColumn(1, "sepal.width", simpleParse)
	r.DefineColumn(2, "petal.length", simpleParse)
	r.DefineColumn(3, "petal.width", simpleParse)
	r.DefineColumn(4, "variety", classParse)

	err := r.ReadTable()

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	variety, err := r.DataTable.Matrix.SliceCol(4)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	isSetosa := make([]float64, r.DataTable.Matrix.Rows)
	isVirginica := make([]float64, r.DataTable.Matrix.Rows)
	isVersicolor := make([]float64, r.DataTable.Matrix.Rows)

	for i, v := range variety {
		switch v {
		case 0:
			isSetosa[i] = 1
			isVirginica[i] = 0
			isVersicolor[i] = 0
		case 1:
			isSetosa[i] = 0
			isVirginica[i] = 1
			isVersicolor[i] = 0
		case 2:
			isSetosa[i] = 0
			isVirginica[i] = 0
			isVersicolor[i] = 1
		default:
			t.Fatalf("failed to parse value %f to flower type", v)
		}
	}

	//lr := 0.1

	//nn := NewNeuralNet(lr, loss.MSE)
}
