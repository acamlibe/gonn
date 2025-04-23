package matrix

import (
	"fmt"
	"strings"
)

type Matrix struct {
	Rows int
	Cols int
	Data []float64
}

func NewMatrix(rows, cols int) (*Matrix, error) {
	if rows < 0 || cols < 0 {
		return nil, fmt.Errorf("failed to create new matrix with dimensions %dx%d",
			rows, cols)
	}

	m := Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]float64, rows*cols),
	}

	return &m, nil
}

func (m *Matrix) At(row, col int) (float64, error) {
	if !m.validIndex(row, col) {
		return 0, fmt.Errorf("index out of bounds [%d,%d] for matrix %dx%d",
			row, col, m.Rows, m.Cols)
	}

	return m.Data[row*m.Cols+col], nil
}

func (m *Matrix) Set(row, col int, s float64) error {
	if !m.validIndex(row, col) {
		return fmt.Errorf("index out of bounds [%d,%d] for matrix %dx%d",
			row, col, m.Rows, m.Cols)
	}

	m.Data[row*m.Cols+col] = s

	return nil
}

func (m *Matrix) ColVec(col int) ([]float64, error) {
	vec := make([]float64, m.Rows)

	for row := range m.Rows {
		v, err := m.At(row, col)

		if err != nil {
			fmt.Errorf("failed to get column vector: %w", err)
		}

		vec[row] = v
	}

	return vec, nil
}

func (m *Matrix) RowVec(row int) ([]float64, error) {
	vec := make([]float64, m.Cols)

	for col := range m.Cols {
		v, err := m.At(row, col)

		if err != nil {
			fmt.Errorf("failed to get row vector: %w", err)
		}

		vec[col] = v
	}

	return vec, nil
}

func (m *Matrix) validIndex(row, col int) bool {
	return row >= 0 && row < m.Rows && col >= 0 && col < m.Cols
}

func (m *Matrix) Multiply(m2 *Matrix) (*Matrix, error) {
	if m.Cols != m2.Rows {
		return nil, fmt.Errorf("cannot multiply: dimensions: %dx%d and %dx%d",
			m.Rows, m.Cols, m2.Rows, m2.Cols)
	}

	result, err := NewMatrix(m.Rows, m2.Cols)

	if err != nil {
		return nil, fmt.Errorf("failed to create new matrix for multiply operation: %w", err)
	}

	for i := range m.Rows {
		for j := range m2.Cols {
			dotp := float64(0)

			for k := range m.Cols {
				dotp += m.Data[i*m.Cols+k] * m2.Data[k*m2.Cols+j]
			}

			result.Data[i*result.Cols+j] = dotp
		}
	}

	return result, nil
}

func (m *Matrix) Transpose() *Matrix {
	result, err := NewMatrix(m.Cols, m.Rows)

	if err != nil {
		panic("creating a new matrix failed during Transpose resulted in fatal error")
	}

	for i := range m.Cols {
		for j := range m.Rows {
			v, err := m.At(j, i)

			if err != nil {
				panic("getting value from matrix resulted in fatal error")
			}

			err = result.Set(i, j, v)

			if err != nil {
				panic("setting value from matrix resulted in fatal error")
			}
		}
	}

	return result
}

func (m *Matrix) String() string {
	var builder strings.Builder

	for i := range m.Rows {
		builder.WriteString("[")
		for j := range m.Cols {
			if j > 0 {
				builder.WriteString(" ")
			}
			builder.WriteString(fmt.Sprintf("%.2f", m.Data[i*m.Cols+j]))
		}
		builder.WriteString("]\n")
	}

	return builder.String()
}
