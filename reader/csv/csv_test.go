package csv

import (
	"errors"
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

func TestRead(t *testing.T) {
	tr := NewReader("test.csv", true)

	tr.DefineColumn(0, "sepal.length", simpleParse)
	tr.DefineColumn(1, "sepal.width", simpleParse)
	tr.DefineColumn(2, "petal.length", simpleParse)
	tr.DefineColumn(3, "petal.width", simpleParse)
	tr.DefineColumn(4, "variety", classParse)

	err := tr.ReadTable()

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	table := tr.DataTable

	if len(table.Cols) != 5 {
		t.Errorf("expected %d column definitions, got %d", 5, len(table.Cols))
	}

	if table.Matrix.Rows != 150 {
		t.Errorf("expected %d matrix rows, got %d", 150, table.Matrix.Rows)
	}

	if table.Matrix.Cols != 5 {
		t.Errorf("expected %d matrix columns, got %d", 5, table.Matrix.Cols)
	}

	t.Logf("table: %+v", table)
}

func TestReadPartial(t *testing.T) {
	tr := NewReader("test.csv", true)

	tr.DefineColumn(0, "sepal.length", simpleParse)
	tr.DefineColumn(2, "petal.length", simpleParse)
	tr.DefineColumn(4, "variety", classParse)

	err := tr.ReadTable()

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	table := tr.DataTable

	if len(table.Cols) != 3 {
		t.Errorf("expected %d column definitions, got %d", 3, len(table.Cols))
	}

	if table.Matrix.Rows != 150 {
		t.Errorf("expected %d matrix rows, got %d", 150, table.Matrix.Rows)
	}

	if table.Matrix.Cols != 3 {
		t.Errorf("expected %d matrix columns, got %d", 3, table.Matrix.Cols)
	}
}

func TestReadOutOfOrder(t *testing.T) {
	tr := NewReader("test.csv", true)

	tr.DefineColumn(0, "sepal.length", simpleParse)
	tr.DefineColumn(3, "sepal.width", simpleParse)
	tr.DefineColumn(1, "petal.length", simpleParse)
	tr.DefineColumn(2, "petal.width", simpleParse)
	tr.DefineColumn(4, "variety", classParse)

	err := tr.ReadTable()

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	table := tr.DataTable

	if len(table.Cols) != 5 {
		t.Errorf("expected %d column definitions, got %d", 5, len(table.Cols))
	}

	if table.Matrix.Rows != 150 {
		t.Errorf("expected %d matrix rows, got %d", 150, table.Matrix.Rows)
	}

	if table.Matrix.Cols != 5 {
		t.Errorf("expected %d matrix columns, got %d", 5, table.Matrix.Cols)
	}

	v, err := table.Matrix.At(0, 1)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if v != float64(0.2) {
		t.Errorf("expected %f, got %f", float64(0.2), v)
	}
}

func TestReadInvalidPath(t *testing.T) {
	tr := NewReader("missing.csv", true)

	tr.DefineColumn(0, "sepal.length", simpleParse)
	tr.DefineColumn(1, "sepal.width", simpleParse)
	tr.DefineColumn(2, "petal.length", simpleParse)
	tr.DefineColumn(3, "petal.width", simpleParse)
	tr.DefineColumn(4, "variety", classParse)

	err := tr.ReadTable()

	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestReadLambdaError(t *testing.T) {
	tr := NewReader("test.csv", true)

	tr.DefineColumn(0, "sepal.length", simpleParse)
	tr.DefineColumn(1, "sepal.width", simpleParse)
	tr.DefineColumn(2, "petal.length", simpleParse)
	tr.DefineColumn(3, "petal.width", simpleParse)
	tr.DefineColumn(4, "variety", func (s *string) (float64, error) {
		return float64(0), errors.New("test error")
	})

	err := tr.ReadTable()

	if err == nil {
		t.Fatal("expected error, got nil")
	}
}
