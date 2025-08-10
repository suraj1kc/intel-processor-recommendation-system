# Intel Processor Recommendation System 🖥️

A content-based recommendation system that helps users find similar Intel processors based on technical specifications. This project demonstrates end-to-end data science workflow from web scraping to building an interactive recommendation application.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)

## 🎯 Project Overview

This project was developed as part of a Data Science course (MDA512) to create a recommendation system for Intel processors. The system uses content-based filtering to suggest similar processors based on their technical specifications.

### Key Objectives:
1. **Data Collection**: Scrape Intel processor specifications from official Intel website
2. **Data Preparation**: Clean and structure the data for analysis
3. **EDA**: Perform exploratory data analysis to understand processor characteristics
4. **Recommendation Algorithm**: Build a similarity-based recommendation engine
5. **Interactive Application**: Create a user-friendly Streamlit interface

## 🔍 Problem Statement

Given the vast array of Intel processors with different specifications, users often struggle to:
- Find processors similar to their current one
- Compare technical specifications across different processor families
- Make informed decisions when upgrading or selecting processors

**Solution**: Build a recommendation system that suggests similar processors based on technical specifications using cosine similarity.

## 📊 Dataset

### Data Source
- **Origin**: Official Intel website processor specifications
- **Collection Method**: Web scraping of CSV files
- **Total Processors**: 148 entries
- **Categories Covered**:
  - Core Processors
  - Core Ultra Processors
  - Xeon Max Processors
  - Xeon Processors

### Data Structure
- **Raw Format**: Multiple CSV files from Intel
- **Processed Format**: Single JSON file with structured data
- **Final Format**: Cleaned CSV with numeric features

### Key Features Used for Recommendations:
- Total cores and threads
- Max turbo frequency (GHz)
- Cache size (MB)
- Base and turbo power (W)
- Max memory size (GB)
- GPU specifications (frequency, execution units)

## 🛠️ Methodology

### 1. Data Collection & Preparation
```
Intel Website → CSV Files → JSON Consolidation → Flattened DataFrame
```

### 2. Data Cleaning Process
- **Unit Removal**: Stripped units (GHz, MB, GB, W, °C, $) from numeric values
- **Type Conversion**: Converted text values to numeric
- **Missing Values**: Filled with median values for numeric features
- **Standardization**: Applied StandardScaler for similarity calculations

### 3. Recommendation Algorithm
- **Approach**: Content-based filtering using cosine similarity
- **Feature Engineering**: Selected 8 key numeric specifications
- **Similarity Metric**: Cosine similarity between standardized feature vectors
- **Output**: Top-N most similar processors with similarity scores

## ✨ Features

### Core Functionality
- **Processor Search**: Find processors by name or browse categories
- **Similarity Calculation**: Cosine similarity-based recommendations
- **Interactive Visualizations**: Compare specifications across processors
- **Detailed Specifications**: View complete technical details

### Web Application Features
- 🔍 **Find Similar Processors**: Enter a processor name to find similar alternatives
- ⚙️ **Filter by Requirements**: Specify your needs (cores, frequency, price, use case)  
- 💰 **Best Value Options**: Find processors that offer the best performance per dollar
- 📊 **Data Explorer**: Visualize and explore the processor dataset

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/suraj1kc/intel-processor-recommendation-system.git
cd intel-processor-recommendation-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify data files**
Ensure these files are present in the `data/` directory:
- `data/intel_processors_features.csv`
- `data/intel_processors_flat.csv`
- `data/intel_processors_master.json`

## 📖 Usage

### Running the Streamlit Application
```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Application Features

#### 🔍 Find Similar Processors
- Search for a processor by name or select from dropdown
- Get AI-powered recommendations based on similarity scores
- View detailed specifications for each recommendation

#### ⚙️ Filter by Requirements  
- Set minimum/maximum cores, frequency, and price
- Filter by use case (Mobile, Desktop, Server, Embedded)
- Filter by category (Core, Core Ultra, Xeon, Xeon Max)
- Sort results by price, performance, or value

#### 💰 Best Value Options
- Set your budget limit
- Find processors with the best performance per dollar
- Compare value scores across different options

#### 📊 Data Explorer
- View dataset statistics and distributions  
- Explore correlations between features
- Browse the raw dataset with search functionality

### Using Jupyter Notebooks

1. **Exploratory Data Analysis**
```bash
jupyter notebook eda.ipynb
```

2. **Recommendation System Development**
```bash
jupyter notebook processor_recommendation_system.ipynb
```

### Example Usage in Python
```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load the processed data
df = pd.read_csv('data/intel_processors_features.csv')

# Initialize and fit scaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.select_dtypes(include=[np.number]))

# Calculate similarity matrix
similarity_matrix = cosine_similarity(scaled_features)

# Get recommendations for a specific processor
def get_recommendations(processor_name, top_n=5):
    # Implementation details in the notebook
    pass
```

## 📁 Project Structure

```
intel-processor-recommendation-system/
│
├── data/                                 # Data files and storage
│   ├── csv/                             # Original CSV files from Intel
│   │   ├── Core_Processors/             # Core processor CSV files
│   │   ├── Core_Ultra_Processors/       # Core Ultra processor CSV files
│   │   ├── Xeon_Max_Processors/         # Xeon Max processor CSV files
│   │   └── Xeon_Processors/             # Xeon processor CSV files
│   ├── intel_processors_master.json     # Raw consolidated data
│   ├── intel_processors_flat.csv        # Flattened data
│   ├── intel_processors_features.csv    # Clean numeric features
│   ├── intel_processors_flat.jsonl      # JSON Lines format
│   ├── intel_processors_flat.parquet    # Optimized storage format
│   └── intel_processors_features.parquet # Clean features in parquet format
│
├── notebooks/                           # Jupyter notebooks
│   ├── eda.ipynb                        # Exploratory Data Analysis
│   └── processor_recommendation_system.ipynb  # Main recommendation system
│
├── streamlit_app.py                     # Web application
├── requirements.txt                     # Python dependencies
└── README.md                           # Project documentation
```

## 🔧 Technical Details

### Data Requirements
The app expects a CSV file named `intel_processors_features.csv` in the `data/` directory with the following columns:
- `processor_name`: Full processor name
- `category`: Processor category 
- `feat.total_cores`: Number of cores
- `feat.total_threads`: Number of threads
- `feat.max_turbo_ghz`: Maximum turbo frequency
- `feat.base_freq_ghz`: Base frequency
- `feat.cache_mb`: Cache size in MB
- `feat.base_power_w`: Base power in watts
- `feat.price_usd`: Price in USD
- `feat.vertical_segment`: Use case (Mobile/Desktop/Server/Embedded)
- Additional feature columns starting with `feat.`

### Data Processing Pipeline
1. **JSON Flattening**: Converted nested JSON structure to tabular format
2. **Feature Selection**: Identified 8 key numeric specifications
3. **Data Cleaning**: Handled missing values and data type conversions
4. **Standardization**: Applied StandardScaler for fair comparison

### Recommendation Algorithm
- **Method**: Content-based filtering
- **Algorithm**: Content-based filtering with cosine similarity
- **Features**: 16 engineered features including performance ratios
- **Scaling**: StandardScaler for feature normalization
- **Visualization**: Interactive charts using Plotly
- **Framework**: Streamlit for web interface
- **Similarity Metric**: Cosine similarity
- **Feature Space**: 8-dimensional numeric feature vector
- **Scalability**: O(n²) similarity matrix computation

### Performance Metrics
- **Dataset Size**: 148 processors × 181 features
- **Cleaned Features**: 8 key specifications
- **Response Time**: < 1 second for recommendations
- **Accuracy**: Based on domain expert validation

## 📈 Results

### Key Findings from EDA
- **Processor Distribution**: Balanced across Core and Xeon families
- **Performance Correlation**: Strong correlation between cores and cache size
- **Power Efficiency**: Inverse relationship between performance and power consumption

### Recommendation Quality
- **Similarity Scores**: Range from 0.0 to 1.0
- **Validation**: Manual verification shows relevant recommendations
- **User Feedback**: Positive reception for processor upgrade suggestions

## 🔮 Future Improvements

### Algorithm Enhancements
- [ ] Implement weighted similarity based on feature importance
- [ ] Add user preference learning capabilities
- [ ] Include price-performance ratio considerations

### Data Expansion
- [ ] Include AMD processors for broader comparison
- [ ] Add real-world performance benchmarks
- [ ] Incorporate user reviews and ratings

### Application Features
- [ ] User account system for saving preferences
- [ ] Advanced filtering and search capabilities
- [ ] Integration with e-commerce platforms

### Technical Improvements
- [ ] Implement caching for faster response times
- [ ] Add API endpoints for external integration
- [ ] Deploy to cloud platform for public access

## 🤝 Contributing

We welcome contributions to improve the Intel Processor Recommendation System!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [suraj1kc](https://github.com/suraj1kc)

## 🙏 Acknowledgments

- Intel Corporation for providing comprehensive processor specifications
- Course instructors and peers for guidance and feedback
- Open source community for excellent Python libraries
- Streamlit team for the amazing web app framework

## 📞 Contact

For questions, suggestions, or collaborations:
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/suraj1kc)
- **Project Issues**: [GitHub Issues](https://github.com/suraj1kc/intel-processor-recommendation-system/issues)

---

**Academic Project**: This system was developed as part of an academic project (MDA512) for analyzing Intel processor specifications and implementing recommendation algorithms. The dataset contains 148 Intel processors across 4 categories with comprehensive feature engineering.

**Note**: This project is for educational purposes and is not affiliated with Intel Corporation. All processor specifications are publicly available data from Intel's official website.
