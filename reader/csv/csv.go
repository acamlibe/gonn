package csv

import (
	"encoding/csv"
	"fmt"
	"gonn/datatable"
	"io"
	"os"
)

type TableReader struct {
	Path      *string
	ColDefs   []ColumnDef
	DataTable *datatable.DataTable
	HasHeader bool
}

type ColumnDef struct {
	Name    *string
	Idx     int
	ParseFn func(v *string) (float64, error)
}

func NewReader(path string, hasHeader bool) *TableReader {
	return &TableReader{
		Path:      &path,
		HasHeader: hasHeader,
	}
}

func (tr *TableReader) DefineColumn(idx int, name string, parseFn func(v *string) (float64, error)) {
	tr.ColDefs = append(tr.ColDefs, ColumnDef{
		Name:    &name,
		Idx:     idx,
		ParseFn: parseFn,
	})
}

func (tr *TableReader) ReadTable() error {
	f, err := os.Open(*tr.Path)

	if err != nil {
		return fmt.Errorf("failed to open file path: %s, with error: %w", *tr.Path, err)
	}

	defer f.Close()

	funcLen := len(tr.ColDefs)

	r := csv.NewReader(f)

	tr.DataTable = datatable.NewDataTable([]string{})

	if tr.HasHeader {
		_, err := r.Read()

		if err != nil {
			return fmt.Errorf("failed to read file header, path: %s, error: %w", *tr.Path, err)
		}

		for _, def := range tr.ColDefs {
			err := tr.DataTable.AddColumn(*def.Name, []float64{})
			if err != nil {
				return fmt.Errorf("failed to add column %s, error: %w", *def.Name, err)
			}
		}
	}

	lineNo := 1

	for {
		line, err := r.Read()

		if err == io.EOF {
			break
		}

		if err != nil {
			panic("csv reader failed reading line")
		}

		row := make([]float64, funcLen)
		lineLen := len(line)

		for i, def := range tr.ColDefs {
			if def.Idx >= lineLen {
				return fmt.Errorf("column index %d out of bounds of parsed line values, got %d (line %d)",
					def.Idx, len(line), lineNo)
			}

			s := line[def.Idx]
			v, err := def.ParseFn(&s)

			if err != nil {
				return fmt.Errorf("failed to parse cell with column parse function")
			}

			row[i] = v
		}

		err = tr.DataTable.AddRow(row)

		if err != nil {
			return fmt.Errorf("failed to add parsed row to datatable: %w", err)
		}

		lineNo++
	}

	return nil
}
