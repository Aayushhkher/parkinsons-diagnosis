import React from 'react';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import App from './App';

const renderWithRouter = (component: React.ReactElement) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  );
};

test('renders dashboard title', () => {
  renderWithRouter(<App />);
  const titleElement = screen.getByText(/Dashboard/i);
  expect(titleElement).toBeInTheDocument();
});

test('renders navigation links', () => {
  renderWithRouter(<App />);
  
  // Check for main navigation links
  expect(screen.getByText(/New Patient/i)).toBeInTheDocument();
  expect(screen.getByText(/Analysis/i)).toBeInTheDocument();
  expect(screen.getByText(/Batch Upload/i)).toBeInTheDocument();
  expect(screen.getByText(/Reports/i)).toBeInTheDocument();
});

test('renders quick actions', () => {
  renderWithRouter(<App />);
  
  // Check for quick action cards
  expect(screen.getByText(/New Patient Assessment/i)).toBeInTheDocument();
  expect(screen.getByText(/Batch Analysis/i)).toBeInTheDocument();
  expect(screen.getByText(/Model Analysis/i)).toBeInTheDocument();
  expect(screen.getByText(/Reports/i)).toBeInTheDocument();
});

test('renders system status', () => {
  renderWithRouter(<App />);
  
  // Check for system status items
  expect(screen.getByText(/Model Status/i)).toBeInTheDocument();
  expect(screen.getByText(/API Status/i)).toBeInTheDocument();
  expect(screen.getByText(/Database Status/i)).toBeInTheDocument();
});

test('renders recent activity', () => {
  renderWithRouter(<App />);
  
  // Check for recent activity items
  expect(screen.getByText(/New Patient Assessment/i)).toBeInTheDocument();
  expect(screen.getByText(/Batch Analysis Complete/i)).toBeInTheDocument();
  expect(screen.getByText(/Model Update/i)).toBeInTheDocument();
}); 