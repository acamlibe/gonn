package datatable

import (
	"testing"
)

func TestNewDataTable(t *testing.T) {
	cols := []string{"a", "b", "c"}

	d := NewDataTable(cols)

	if d.Matrix.Rows != 0 {
		t.Errorf("expected %d rows, got %d", 0, d.Matrix.Rows)
	}

	if d.Matrix.Cols != len(cols) {
		t.Errorf("expected %d columns, got %d", len(cols), d.Matrix.Cols)
	}
}

func TestAddRow(t *testing.T) {
	cols := []string{"a", "b", "c"}

	d := NewDataTable(cols)

	err := d.AddRow([]float64{1, 2, 3})

	if err != nil {
		t.Fatalf("expected no error, got error: %v", err)
	}

	err = d.AddRow([]float64{4, 5, 6})

	if err != nil {
		t.Fatalf("expected no error, got error: %v", err)
	}

	if d.Matrix.Rows != 2 {
		t.Errorf("expected %d rows, got %d", 2, d.Matrix.Rows)
	}

	if d.Matrix.Cols != 3 {
		t.Errorf("expected %d columns, got %d", 3, d.Matrix.Cols)
	}

	if len(d.Matrix.Data) != 6 {
		t.Errorf("expected %d data length, got %d", 6, len(d.Matrix.Data))
	}

	if d.Cols[1] != "b" {
		t.Errorf("expected %s as 2nd column, got %s", "b", d.Cols[1])
	}

	v, err := d.Matrix.At(1, 1)

	if err != nil {
		t.Fatalf("expected no error, got error: %v", err)
	}

	if v != 5 {
		t.Errorf("expected %f value for index [%d,%d], got %f", float64(5), 1, 1, v)
	}
}

func TestAddRowError(t *testing.T) {
	cols := []string{"a", "b", "c"}

	d := NewDataTable(cols)

	err := d.AddRow([]float64{1, 2, 3, 4})

	if err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestAddColumn(t *testing.T) {
	cols := []string{"a", "b", "c"}

	d := NewDataTable(cols)

	err := d.AddRow([]float64{1, 2, 3})

	if err != nil {
		t.Fatalf("expected no error, got error: %v", err)
	}

	err = d.AddRow([]float64{4, 5, 6})

	if err != nil {
		t.Fatalf("expected no error, got error: %v", err)
	}

	err = d.AddColumn("d", []float64{7, 8})

	if err != nil {
		t.Fatalf("expected no error, got error: %v", err)
	}

	if len(d.Matrix.Data) != 8 {
		t.Errorf("expected %d rows, but got %d", 8, len(d.Matrix.Data))
	}

	s, err := d.Matrix.At(1, 3)

	if err != nil {
		t.Fatalf("expected no error, got error: %v", err)
	}

	if s != float64(8) {
		t.Errorf("expected %f, got %f", float64(8), s)
	}
}

func TestAddColumnError(t *testing.T) {
	cols := []string{"a", "b", "c"}

	d := NewDataTable(cols)

	err := d.AddColumn("d", []float64{4})

	if err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestAddColumnEmpty(t *testing.T) {
	cols := []string{"a", "b", "c"}

	d := NewDataTable(cols)

	err := d.AddColumn("d", []float64{})

	if err != nil {
		t.Fatalf("expected no error, got error: %v", err)
	}

	if len(d.Matrix.Data) > 0 {
		t.Errorf("expected %d rows, but got %d", 0, len(d.Matrix.Data))
	}
}

func TestAddColumnNoColumns(t *testing.T) {
	d := NewDataTable([]string{})

	err := d.AddColumn("a", []float64{})

	if err != nil {
		t.Fatalf("expected no error, got error: %v", err)
	}

	if len(d.Matrix.Data) > 0 {
		t.Errorf("expected %d rows, but got %d", 0, len(d.Matrix.Data))
	}
}

func TestRemoveColumn(t *testing.T) {
	cols := []string{"a", "b", "c"}
	colToRemove := "c"

	d := NewDataTable(cols)

	err := d.AddRow([]float64{1, 2, 3})

	if err != nil {
		t.Fatalf("expected no error, got error: %v", err)
	}

	err = d.AddRow([]float64{4, 5, 6})

	if err != nil {
		t.Fatalf("expected no error, got error: %v", err)
	}

	err = d.RemoveColumn(colToRemove)

	if err != nil {
		t.Fatalf("expected no error, got error: %v", err)
	}

	if len(d.Cols) != 2 {
		t.Errorf("expected %d datatable columns, got %d", 2, len(d.Cols))
	}

	if d.Matrix.Cols != 2 {
		t.Errorf("expected %d matrix columns, got %d", 2, d.Matrix.Cols)
	}

	if len(d.Matrix.Data) != 4 {
		t.Errorf("expected %d matrix data length, got %d", 4, len(d.Matrix.Data))
	}

	v, err := d.Matrix.At(0, 1)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if v != float64(2) {
		t.Errorf("expected %f matrix value for (%d,%d), got %f", float64(2), 0, 1, v)
	}
}