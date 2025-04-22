package matrix

import (
	"testing"
)

func TestNewMatrix(t *testing.T) {
    rows := 2
    cols := 3

    m, err := NewMatrix(rows, cols)

    if err != nil {
        t.Fatalf("expected no error, got: %v", err)
    }

    if m == nil {
        t.Fatalf("expected Matrix, got nil")
    }

    if len(m.Data) != rows*cols {
        t.Errorf("expected Data length %d, got %d", rows*cols, len(m.Data))
    }

    if !(m.Rows == rows && m.Cols == cols) {
        t.Errorf("expected dimensions %dx%d, got %dx%d", rows, cols, m.Rows, m.Cols)
    }
}

func TestNewMatrixError(t *testing.T) {
    rows := -1
    cols := 3

    _, err := NewMatrix(rows, cols)

    if err == nil {
        t.Fatalf("expected error, but got nil")
    }
}

func TestMultiplyMatrices(t *testing.T) {
    // Input M1
    // [2 3 4]
    // [5 6 7]

    // Input M2
    // [5 3]
    // [2 8]
    // [9 7]

    // Output M
    // [52 58]
    // [100 112]

    m1, _ := NewMatrix(2, 3)
    _ = m1.Set(0, 0, 2)
    _ = m1.Set(0, 1, 3)
    _ = m1.Set(0, 2, 4)

    _ = m1.Set(1, 0, 5)
    _ = m1.Set(1, 1, 6)
    _ = m1.Set(1, 2, 7)

    m2, _ := NewMatrix(3, 2)
    _ = m2.Set(0, 0, 5)
    _ = m2.Set(0, 1, 3)

    _ = m2.Set(1, 0, 2)
    _ = m2.Set(1, 1, 8)

    _ = m2.Set(2, 0, 9)
    _ = m2.Set(2, 1, 7)

    m, err := m1.Multiply(m2)

    exp := make([]float64, 4)
    exp[0] = 52
    exp[1] = 58
    exp[2] = 100
    exp[3] = 112

    if err != nil {
        t.Fatalf("expected no error, got: %v", err)
    }

    if len(exp) != len(m.Data) {
        t.Errorf("expected Data length %d, got %d", len(exp), len(m.Data))
    }

    if v, _ := m.At(0, 0); v != exp[0] {
        t.Errorf("expected 1x1 to be %f, but got %f", exp[0], v)
    }

    if v, _ := m.At(0, 1); v != exp[1] {
        t.Errorf("expected 1x2 to be %f, but got %f", exp[1], v)
    }

    if v, _ := m.At(1, 0); v != exp[2] {
        t.Errorf("expected 2x1 to be %f, but got %f", exp[2], v)
    }

    if v, _ := m.At(1, 1); v != exp[3] {
        t.Errorf("expected 2x2 to be %f, but got %f", exp[3], v)
    }
}

func TestMultiplyMatricesError(t *testing.T) {
    // Input M1
    // [2 3 4]
    // [5 6 7]

    // Input M2
    // [5 3 7]
    // [2 8 9]

    // Rule: m x n * n x p

    m1, _ := NewMatrix(2, 3)
    _ = m1.Set(0, 0, 2)
    _ = m1.Set(0, 1, 3)
    _ = m1.Set(0, 2, 4)

    _ = m1.Set(1, 0, 5)
    _ = m1.Set(1, 1, 6)
    _ = m1.Set(1, 2, 7)

    m2, _ := NewMatrix(2, 3)
    _ = m2.Set(0, 0, 5)
    _ = m2.Set(0, 1, 3)
    _ = m2.Set(0, 1, 7)

    _ = m2.Set(1, 0, 2)
    _ = m2.Set(1, 1, 8)
    _ = m2.Set(1, 1, 9)

    _, err := m1.Multiply(m2)

    if err == nil {
        t.Errorf("expected error, got nil")
    }
}

func TestTranspose(t *testing.T) {
    // Input M1
    // [2 3 4]
    // [5 6 7]

    // Output M
    // [2 5]
    // [3 6]
    // [4 7]

    m1, _ := NewMatrix(2, 3)
    _ = m1.Set(0, 0, 2)
    _ = m1.Set(0, 1, 3)
    _ = m1.Set(0, 2, 4)

    _ = m1.Set(1, 0, 5)
    _ = m1.Set(1, 1, 6)
    _ = m1.Set(1, 2, 7)

    m := m1.Transpose()

    exp := make([]float64, 6)
    exp[0] = 2
    exp[1] = 5
    exp[2] = 3
    exp[3] = 6
    exp[4] = 4
    exp[5] = 7

    if !(m.Rows == m1.Cols && m.Cols == m1.Rows) {
        t.Errorf("expected dimensions %dx%d, got %dx%d", 3, 2, m.Rows, m.Cols)
    }


    if v, _ := m.At(0, 0); v != exp[0] {
        t.Errorf("expected 1x1 to be %f, but got %f", exp[0], v)
    }

    if v, _ := m.At(0, 1); v != exp[1] {
        t.Errorf("expected 1x2 to be %f, but got %f", exp[1], v)
    }

    if v, _ := m.At(1, 0); v != exp[2] {
        t.Errorf("expected 2x1 to be %f, but got %f", exp[2], v)
    }

    if v, _ := m.At(1, 1); v != exp[3] {
        t.Errorf("expected 2x2 to be %f, but got %f", exp[3], v)
    }

    if v, _ := m.At(2, 0); v != exp[4] {
        t.Errorf("expected 3x1 to be %f, but got %f", exp[4], v)
    }

    if v, _ := m.At(2, 1); v != exp[5] {
        t.Errorf("expected 3x2 to be %f, but got %f", exp[5], v)
    }
}