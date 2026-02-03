# Agent Tools Viewer

A clean web interface for viewing agent tool definitions from JSON files.

## Usage

### Option 1: Local Server (Recommended)
```bash
cd viewer
python3 -m http.server 8000
```
Then open http://localhost:8000 in your browser.

### Option 2: File Selection
Open `index.html` directly in your browser and use the file selector to choose a JSON file.

## Features

- Clean, collapsible tool cards
- Search/filter by tool name or description
- View required parameters at a glance
- Expandable full descriptions and schemas
- Light mode design

## JSON Format

Place your tools JSON files in this directory. The viewer expects an array of tool objects with:
- `name`: Tool name
- `description`: Tool description
- `input_schema`: JSON schema for tool parameters
