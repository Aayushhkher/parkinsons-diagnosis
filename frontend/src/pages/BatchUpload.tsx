import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import axios from 'axios';

interface BatchResult {
  status: string;
  predictions: {
    [key: string]: number[];
  };
  processed_rows: number;
}

const BatchUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BatchResult | null>(null);
  const [previewData, setPreviewData] = useState<any[]>([]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      setResult(null);
      previewFile(selectedFile);
    }
  };

  const previewFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = e.target?.result as string;
        const rows = data.split('\n').slice(1); // Skip header
        const preview = rows.slice(0, 5).map(row => {
          const values = row.split(',');
          return {
            id: values[0],
            age: values[1],
            gender: values[2],
            duration: values[3],
            motor_UPDRS: values[4],
            total_UPDRS: values[5],
            tremor: values[6],
            rigidity: values[7],
            bradykinesia: values[8],
          };
        });
        setPreviewData(preview);
      } catch (err) {
        setError('Error reading file. Please check the file format.');
        console.error(err);
      }
    };
    reader.readAsText(file);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/upload-data', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err) {
      setError('Error uploading file. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const columns: GridColDef[] = [
    { field: 'id', headerName: 'ID', width: 90 },
    { field: 'age', headerName: 'Age', width: 90 },
    { field: 'gender', headerName: 'Gender', width: 90 },
    { field: 'duration', headerName: 'Duration', width: 90 },
    { field: 'motor_UPDRS', headerName: 'Motor UPDRS', width: 120 },
    { field: 'total_UPDRS', headerName: 'Total UPDRS', width: 120 },
    { field: 'tremor', headerName: 'Tremor', width: 90 },
    { field: 'rigidity', headerName: 'Rigidity', width: 90 },
    { field: 'bradykinesia', headerName: 'Bradykinesia', width: 120 },
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Batch Patient Analysis
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ mb: 3 }}>
          <input
            accept=".csv"
            style={{ display: 'none' }}
            id="batch-file-upload"
            type="file"
            onChange={handleFileChange}
          />
          <label htmlFor="batch-file-upload">
            <Button
              variant="contained"
              component="span"
              sx={{ mr: 2 }}
            >
              Select File
            </Button>
            {file && file.name}
          </label>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {file && (
          <>
            <Typography variant="h6" gutterBottom>
              File Preview
            </Typography>
            <Box sx={{ height: 300, width: '100%', mb: 3 }}>
              <DataGrid
                rows={previewData}
                columns={columns}
                pageSize={5}
                rowsPerPageOptions={[5]}
                disableSelectionOnClick
              />
            </Box>

            <Button
              variant="contained"
              onClick={handleUpload}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Upload and Analyze'}
            </Button>
          </>
        )}
      </Paper>

      {result && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Analysis Results
          </Typography>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Model</TableCell>
                  <TableCell>PD Positive</TableCell>
                  <TableCell>PD Negative</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(result.predictions).map(([model, predictions]) => (
                  <TableRow key={model}>
                    <TableCell>{model.toUpperCase()}</TableCell>
                    <TableCell>
                      {predictions.filter(p => p === 1).length}
                    </TableCell>
                    <TableCell>
                      {predictions.filter(p => p === 0).length}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          <Typography variant="body1" sx={{ mt: 2 }}>
            Total records processed: {result.processed_rows}
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default BatchUpload; 