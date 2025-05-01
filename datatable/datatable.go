package datatable

import (
	"fmt"
	"gonn/matrix"
	"strconv"
	"strings"
)

type DataTable struct {
	Matrix *matrix.Matrix
	Cols   []string
}

func NewDataTable(cols []string) *DataTable {
	m, err := matrix.NewMatrix(0, len(cols))

	if err != nil {
		panic("failed to create datatable matrix")
	}

	d := DataTable{
		Matrix: m,
		Cols:   cols,
	}

	return &d
}

func (d *DataTable) AddRow(data []float64) error {
	if len(d.Cols) != len(data) {
		return fmt.Errorf("mismatch between data length and datatable column length")
	}

	for _, s := range data {
		d.Matrix.Data = append(d.Matrix.Data, s)
	}

	d.Matrix.Rows++

	return nil
}

func (d *DataTable) AddColumn(name string, data []float64) error {
	m := d.Matrix

	if m.Rows != len(data) {
		return fmt.Errorf("mismatch between datatable row count and data length")
	}

	m.Cols++
	d.Cols = append(d.Cols, name)

	for i, s := range data {
		m.Data = append(m.Data, 0)
		mIdx := i*m.Cols + (m.Cols - 1)

		copy(m.Data[mIdx+1:], m.Data[mIdx:])

		m.Data[mIdx] = s
	}

	return nil
}

func (d *DataTable) RemoveColumn(name string) error {
	// Find the index of the column to remove
	colIdx, err := d.findColIdx(name)
	if err != nil {
		return err
	}

	// Remove the column name from Cols slice
	d.Cols = append(d.Cols[:colIdx], d.Cols[colIdx+1:]...)

	// Remove the column data from the matrix
	m := d.Matrix
	newData := make([]float64, 0, (m.Cols-1)*m.Rows)

	for row := 0; row < m.Rows; row++ {
		for col := 0; col < m.Cols; col++ {
			if col == colIdx {
				continue // Skip the column we're removing
			}
			idx := row*m.Cols + col
			newData = append(newData, m.Data[idx])
		}
	}

	// Update the matrix
	m.Data = newData
	m.Cols--

	return nil
}

func (d *DataTable) findColIdx(name string) (int, error) {
	for i, col := range d.Cols {
		if strings.EqualFold(col, name) {
			return i, nil
		}
	}

	return 0, fmt.Errorf("failed to find column: %v", name)
}

func (d *DataTable) String() string {
	var builder strings.Builder

	builder.WriteString("\n[")
	for i, c := range d.Cols {
		if i > 0 {
			builder.WriteString(" ")
		}

		builder.WriteString(c)
	}
	builder.WriteString("]\n")

	for row := range d.Matrix.Rows {
		if row >= 100 {
			break
		}

		builder.WriteString("[")
		for col := range d.Matrix.Cols {
			if col >= 10 {
				break
			}

			if col > 0 {
				builder.WriteString(" ")
			}

			v, err := d.Matrix.At(row, col)

			if err != nil {
				panic("failed to get value of matrix for String print method")
			}

			builder.WriteString(strconv.FormatFloat(v, 'f', 2, 64))
		}
		builder.WriteString("]\n")
	}

	return builder.String()
}
