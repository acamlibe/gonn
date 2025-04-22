package datatable

import (
	"fmt"
	"gonn/matrix"
)

type DataTable struct {
	Matrix *matrix.Matrix
	Cols []string
}

func NewDataTable(cols []string) (*DataTable, error) {
	m, err := matrix.NewMatrix(1, len(cols))

	if err != nil {
		return nil, fmt.Errorf("failed to create data table: %w", err)
	}

	d := DataTable{
		Matrix: m,
		Cols: cols,
	}

	return &d, nil
}

func (d *DataTable) AddRow(data []float64) error {
	if len(d.Cols) != len(data) {
		return fmt.Errorf("mismatch between data length and datatable column length")
	}

	for _, s := range data {
		d.Matrix.Data = append(d.Matrix.Data, s)
	}

	return nil
}

func (d *DataTable) AddColumn(name string, data []float64) error {
	m := d.Matrix

	if m.Rows != len(data) {
		return fmt.Errorf("mismatch between datatable row count and data length")
	}

	for i, s := range data {
		m.Data = append(m.Data, 0)
		mIdx := i * m.Cols

		copy(m.Data[mIdx+1:], m.Data[mIdx:])

		m.Data[mIdx] = s
	}

	m.Cols++
	d.Cols = append(d.Cols, name)

	return nil
}