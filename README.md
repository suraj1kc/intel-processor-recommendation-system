# Interactive Intel Processor Recommendation System

## 🚀 Features

### 1. 🎯 Personalized Recommendations
- **Processor Type Filtering**: Search by family ('Core 3', 'Core 5', 'i7', 'Xeon')
- **Price Range Selection**: Budget categories or custom ranges
- **Usage-Based Filtering**: Gaming, Content Creation, Office Work, Programming, Server
- **Performance Priority**: Single-core, Multi-core, Power efficiency, or Balanced

### 2. 🔍 Smart Search
- **Short Search Terms**: Use 'Core 5' instead of full processor names
- **Examples Provided**: Clear examples for each processor family
- **Price-Organized Results**: Grouped by budget categories
- **Quick Specs Display**: Cores, threads, frequency, price at a glance

### 3. 💰 Advanced Price Filtering
- **Budget Categories**:
  - 💚 Budget: $0 - $300
  - 💙 Mid-range: $300 - $600
  - 💜 High-end: $600 - $1,000
  - 🧡 Premium: $1,000 - $2,000
  - ❤️ Ultra Premium: $2,000+
- **Custom Range**: Set your own min/max budget
- **Value Ratings**: Performance-per-dollar calculations

### 4. 🔄 Processor Comparison
- **Side-by-Side Comparison**: Compare any two processors
- **Color-Coded Winners**: Green/red indicators for better specs
- **Value Analysis**: Which offers better price/performance
- **Recommendation Engine**: Smart suggestions based on comparison

### 5. 📊 Detailed Specifications
- **Complete Specs**: All technical details in one view
- **Performance Metrics**: Efficiency calculations
- **Usage Recommendations**: Best use cases for each processor

## 🎮 Usage Examples

### Quick Search Terms:
```
'Core 3'     → Entry-level processors
'Core 5'     → Mid-range processors  
'Core 7'     → High-performance processors
'Core 9'     → Flagship processors
'Core Ultra' → Premium processors
'Xeon'       → Server/workstation processors
'i5', 'i7'   → Classic Intel naming
'13700'      → Specific model numbers
```

### Price Range Examples:
```
Gaming Build:      $300-600 (Core 5/7)
Content Creation:  $600-1000 (Core 7/9, Multi-core focus)
Office Work:       $0-300 (Core 3/5, Efficiency focus)
Server/Enterprise: $1000+ (Xeon processors)
```

## 🚀 How to Run

### Option 1: Full Interactive Experience
```bash
python interactive_recommend.py
```
- Complete menu system
- All features available
- User-friendly interface

### Option 2: Simple Recommendations
```bash
python recommend.py
```
- Quick search and recommendations
- Similarity-based suggestions
- Simpler interface

### Option 3: View Demo
```bash
python demo.py
```
- Overview of available features
- Database statistics
- Search examples

## 📋 Main Menu Options

1. **🎯 Personalized Recommendations**
   - Answer preference questions
   - Get filtered recommendations
   - See performance scores and value ratings

2. **🔍 Search by Processor Name**
   - Use short search terms
   - Browse by price categories
   - View detailed specifications

3. **📊 Browse by Category**
   - Core Processors
   - Core Ultra Processors
   - Xeon Processors
   - Xeon Max Processors

4. **🔄 Compare Processors**
   - Side-by-side comparison
   - Performance analysis
   - Value recommendations

## 🎯 Sample Interaction Flow

```
1. Select "Personalized Recommendations"
2. Enter processor type: "Core 5"
3. Choose budget: "Mid-range ($300-600)"
4. Select usage: "Gaming"
5. Choose priority: "Single-core performance"
6. Get top 5 filtered recommendations with:
   - Price and specs
   - Performance scores
   - Value ratings
   - Usage recommendations
```

## 📊 Database Info

- **148 Total Processors**
- **Price Range**: $134 - $19,000
- **Categories**: Core, Core Ultra, Xeon, Xeon Max
- **Features**: 15+ technical specifications per processor

## 💡 Tips for Best Results

1. **Use Short Search Terms**: 'Core 5' works better than full names
2. **Set Realistic Budgets**: Use price categories for better filtering
3. **Consider Your Usage**: Different tasks need different processor strengths
4. **Compare Options**: Use comparison feature for final decisions
5. **Check Value Ratings**: Balance performance with price

## 🔧 Technical Features

- **Machine Learning**: Cosine similarity for recommendations
- **Smart Filtering**: Multi-criteria processor selection
- **Performance Scoring**: Custom algorithms based on usage
- **Value Analysis**: Price-performance calculations
- **Interactive UI**: Clear, emoji-enhanced interface
