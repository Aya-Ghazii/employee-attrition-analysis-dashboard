import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="نظام تحليل ترك الموظفين - Employee Attrition Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Arabic support and professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .arabic-text {
        font-family: 'Cairo', 'Arial', sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    .insight-card {
        background: linear-gradient(145deg, #e3f2fd, #f8f9fa);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }
    
    .warning-card {
        background: linear-gradient(145deg, #fff3e0, #fafafa);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
    }
    
    .success-card {
        background: linear-gradient(145deg, #e8f5e8, #f8f9fa);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #ffffff);
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .recommendations-card {
        background: linear-gradient(145deg, #f3e5f5, #fafafa);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #9c27b0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EmployeeAttritionAnalyzer:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.insights = []
        # Define consistent color scheme
        self.color_scheme = {'ذكر': '#1E88E5', 'أنثى': '#E91E63'}  # Blue for male, Pink for female
        
    def load_sample_data(self):
        """Load comprehensive sample data for demonstration"""
        np.random.seed(42)
        
        # Arabic reason mappings
        reasons_arabic = [
            'فرصة عمل أخرى', 'راتب غير مناسب', 'بيئة عمل سيئة', 'عدم الترقية',
            'ساعات عمل طويلة', 'التفرغ للعائلة', 'أسباب شخصية', 'عدم الرضا الوظيفي',
            'مشاكل مع الإدارة', 'نقل لمكان آخر', 'ظروف صحية', 'استكمال الدراسة',
            'تغيير المسار المهني', 'عدم الاستقرار الوظيفي'
        ]
        
        departments_arabic = [
            'الموارد البشرية', 'المالية والمحاسبة', 'التسويق', 'المبيعات',
            'تقنية المعلومات', 'العمليات', 'خدمة العملاء', 'الإنتاج',
            'البحث والتطوير', 'الجودة'
        ]
        
        # Generate 2000 records across multiple years
        n_records = 2000
        
        sample_data = {
            'رقم_الموظف': range(1, n_records + 1),
            'الجنس': np.random.choice(['ذكر', 'أنثى'], n_records, p=[0.6, 0.4]),
            'السن': np.random.normal(35, 8, n_records).astype(int),
            'الإدارة': np.random.choice(departments_arabic, n_records),
            'السبب': np.random.choice(reasons_arabic, n_records, 
                                     p=[0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.06, 0.06, 
                                        0.05, 0.05, 0.04, 0.04, 0.05, 0.05]),
            'سنة_الترك': np.random.choice(range(2015, 2025), n_records,
                                        p=[0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.12, 0.11, 0.09, 0.05]),
            'مدة_الخدمة_بالسنوات': np.random.exponential(3, n_records),
            'الراتب_الشهري': np.random.normal(8000, 2500, n_records)
        }
        
        # Clean the data
        sample_data['السن'] = np.clip(sample_data['السن'], 22, 65)
        sample_data['مدة_الخدمة_بالسنوات'] = np.round(np.clip(sample_data['مدة_الخدمة_بالسنوات'], 0.5, 20), 1)
        sample_data['الراتب_الشهري'] = np.round(np.clip(sample_data['الراتب_الشهري'], 3000, 25000), 0)
        
        return pd.DataFrame(sample_data)
    
    def process_data(self, df):
        """Process and clean the data"""
        self.data = df.copy()
        
        # Create age groups
        self.data['فئة_العمر'] = pd.cut(self.data['السن'], 
                                       bins=[0, 25, 35, 45, 55, 100], 
                                       labels=['أقل من 25', '25-35', '35-45', '45-55', 'أكثر من 55'])
        
        # Create service duration groups
        self.data['فئة_مدة_الخدمة'] = pd.cut(self.data['مدة_الخدمة_بالسنوات'],
                                           bins=[0, 1, 3, 5, 10, 100],
                                           labels=['أقل من سنة', '1-3 سنوات', '3-5 سنوات', '5-10 سنوات', 'أكثر من 10 سنوات'])
        
        # Create salary groups
        self.data['فئة_الراتب'] = pd.cut(self.data['الراتب_الشهري'],
                                        bins=[0, 5000, 8000, 12000, 20000, 100000],
                                        labels=['أقل من 5000', '5000-8000', '8000-12000', '12000-20000', 'أكثر من 20000'])
        
        self.processed_data = self.data.copy()
        self.generate_insights()
        
        return True
    
    def generate_insights(self):
        """Generate automated insights from the data"""
        if self.processed_data is None:
            return
        
        insights = []
        
        # Gender insights
        male_pct = (self.processed_data['الجنس'] == 'ذكر').mean() * 100
        female_pct = (self.processed_data['الجنس'] == 'أنثى').mean() * 100
        
        if male_pct > 65:
            insights.append(f"🚨 معدل ترك الذكور مرتفع جداً ({male_pct:.1f}%)")
        elif female_pct > 50:
            insights.append(f"⚠️ معدل ترك الإناث أعلى من المتوقع ({female_pct:.1f}%)")
        
        # Top reasons
        top_reason = self.processed_data['السبب'].mode().iloc[0]
        top_reason_pct = (self.processed_data['السبب'] == top_reason).mean() * 100
        insights.append(f"📊 السبب الأكثر شيوعاً: {top_reason} ({top_reason_pct:.1f}%)")
        
        # Yearly trends
        yearly_trend = self.processed_data.groupby('سنة_الترك').size()
        if len(yearly_trend) > 1:
            recent_years = yearly_trend.iloc[-3:].mean()
            earlier_years = yearly_trend.iloc[:-3].mean()
            if recent_years > earlier_years * 1.2:
                insights.append("📈 اتجاه تصاعدي واضح في معدل الاستقالات خلال السنوات الأخيرة")
            elif recent_years < earlier_years * 0.8:
                insights.append("📉 انخفاض ملحوظ في معدل الاستقالات مؤخراً")
        
        # Department insights
        dept_counts = self.processed_data['الإدارة'].value_counts()
        highest_turnover_dept = dept_counts.index[0]
        insights.append(f"🏢 أعلى معدل ترك في إدارة: {highest_turnover_dept}")
        
        # Age insights
        young_employees = (self.processed_data['السن'] < 30).mean() * 100
        if young_employees > 40:
            insights.append(f"👶 نسبة عالية من الموظفين الشباب يتركون العمل ({young_employees:.1f}%)")
        
        # Service duration insights
        short_service = (self.processed_data['مدة_الخدمة_بالسنوات'] < 2).mean() * 100
        if short_service > 30:
            insights.append(f"⏰ {short_service:.1f}% من المستقيلين خدموا أقل من سنتين")
        
        self.insights = insights
    
    def get_summary_stats(self):
        """Get summary statistics"""
        if self.processed_data is None:
            return {}
        
        total = len(self.processed_data)
        male_count = (self.processed_data['الجنس'] == 'ذكر').sum()
        female_count = (self.processed_data['الجنس'] == 'أنثى').sum()
        
        avg_age = self.processed_data['السن'].mean()
        avg_service = self.processed_data['مدة_الخدمة_بالسنوات'].mean()
        avg_salary = self.processed_data['الراتب_الشهري'].mean()
        
        years_range = f"{self.processed_data['سنة_الترك'].min()} - {self.processed_data['سنة_الترك'].max()}"
        
        return {
            'total': total,
            'male_count': male_count,
            'female_count': female_count,
            'male_pct': (male_count / total * 100) if total > 0 else 0,
            'female_pct': (female_count / total * 100) if total > 0 else 0,
            'avg_age': avg_age,
            'avg_service': avg_service,
            'avg_salary': avg_salary,
            'years_range': years_range,
            'unique_reasons': self.processed_data['السبب'].nunique(),
            'unique_departments': self.processed_data['الإدارة'].nunique()
        }
    
    def generate_recommendations(self):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Top reason recommendations
        top_reason = self.processed_data['السبب'].mode().iloc[0]
        if top_reason == 'فرصة عمل أخرى':
            recommendations.append("💼 تحسين حزمة المزايا والرواتب للاحتفاظ بالمواهب")
            recommendations.append("🎯 تطوير برامج الاحتفاظ بالموظفين المتميزين")
        elif top_reason == 'راتب غير مناسب':
            recommendations.append("💰 مراجعة هيكل الرواتب ومقارنته بالسوق")
            recommendations.append("📊 إجراء دراسة رواتب شاملة")
        elif top_reason == 'بيئة عمل سيئة':
            recommendations.append("🏢 تحسين بيئة العمل والثقافة المؤسسية")
            recommendations.append("🤝 تعزيز التواصل بين الإدارة والموظفين")
        
        # Service duration recommendations
        short_service_pct = (self.processed_data['مدة_الخدمة_بالسنوات'] < 2).mean() * 100
        if short_service_pct > 30:
            recommendations.append("🎓 تطوير برامج توجيه أفضل للموظفين الجدد")
            recommendations.append("📋 مراجعة عملية التوظيف والاختيار")
        
        # Gender-specific recommendations
        male_pct = (self.processed_data['الجنس'] == 'ذكر').mean() * 100
        if male_pct > 65:
            recommendations.append("👨‍💼 التركيز على احتياجات الموظفين الذكور")
        else:
            recommendations.append("👩‍💼 تطوير برامج دعم التوازن بين العمل والحياة")
        
        # Department recommendations
        top_dept = self.processed_data['الإدارة'].value_counts().index[0]
        recommendations.append(f"🎯 إيلاء اهتمام خاص لإدارة {top_dept}")
        
        return recommendations

def create_reason_analysis_chart(data):
    """Create comprehensive reason analysis chart"""
    reason_counts = data['السبب'].value_counts().head(10)
    
    fig = px.bar(
        x=reason_counts.values,
        y=reason_counts.index,
        orientation='h',
        title="أهم أسباب ترك العمل",
        labels={'x': 'عدد الحالات', 'y': 'السبب'},
        color=reason_counts.values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=500,
        font=dict(size=12),
        title_x=0.5,
        showlegend=False
    )
    
    return fig

def create_gender_reason_heatmap(data):
    """Create heatmap showing reasons by gender"""
    pivot_data = data.groupby(['السبب', 'الجنس']).size().unstack(fill_value=0)
    
    fig = px.imshow(
        pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        title="خريطة حرارية: أسباب الترك حسب الجنس",
        labels={'color': 'عدد الحالات'},
        aspect='auto'
    )
    
    fig.update_layout(height=600)
    return fig

def create_yearly_trend_chart(data, color_scheme):
    """Create yearly trend analysis with blue and pink colors"""
    yearly_data = data.groupby(['سنة_الترك', 'الجنس']).size().reset_index(name='العدد')
    
    fig = px.line(
        yearly_data,
        x='سنة_الترك',
        y='العدد',
        color='الجنس',
        title="اتجاه الاستقالات عبر السنوات حسب الجنس",
        markers=True,
        color_discrete_map=color_scheme
    )
    
    fig.update_layout(
        xaxis_title="السنة",
        yaxis_title="عدد الاستقالات",
        height=400
    )
    
    return fig

def create_attrition_line_graph(data, color_scheme):
    """Create a new line graph showing attrition trends by gender over time"""
    # Group by year and gender to get counts
    line_data = data.groupby(['سنة_الترك', 'الجنس']).size().reset_index(name='عدد_الاستقالات')
    
    # Create line graph
    fig = go.Figure()
    
    # Add line for males
    male_data = line_data[line_data['الجنس'] == 'ذكر']
    fig.add_trace(go.Scatter(
        x=male_data['سنة_الترك'],
        y=male_data['عدد_الاستقالات'],
        mode='lines+markers',
        name='ذكر',
        line=dict(color=color_scheme['ذكر'], width=3),
        marker=dict(size=8, color=color_scheme['ذكر'])
    ))
    
    # Add line for females
    female_data = line_data[line_data['الجنس'] == 'أنثى']
    fig.add_trace(go.Scatter(
        x=female_data['سنة_الترك'],
        y=female_data['عدد_الاستقالات'],
        mode='lines+markers',
        name='أنثى',
        line=dict(color=color_scheme['أنثى'], width=3),
        marker=dict(size=8, color=color_scheme['أنثى'])
    ))
    
    fig.update_layout(
        title="اتجاه الاستقالات عبر السنوات - مخطط خطي مفصل",
        xaxis_title="السنة",
        yaxis_title="عدد الاستقالات",
        height=450,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_department_analysis(data, color_scheme):
    """Create department analysis chart with consistent colors"""
    dept_gender = data.groupby(['الإدارة', 'الجنس']).size().reset_index(name='العدد')
    
    fig = px.bar(
        dept_gender,
        x='الإدارة',
        y='العدد',
        color='الجنس',
        title="الاستقالات حسب الإدارة والجنس",
        color_discrete_map=color_scheme
    )
    
    fig.update_layout(
        xaxis_title="الإدارة",
        yaxis_title="عدد الاستقالات",
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig

def create_service_duration_analysis(data, color_scheme):
    """Analyze attrition by service duration with consistent colors"""
    service_data = data.groupby(['فئة_مدة_الخدمة', 'الجنس']).size().reset_index(name='العدد')
    
    fig = px.bar(
        service_data,
        x='فئة_مدة_الخدمة',
        y='العدد',
        color='الجنس',
        title="الاستقالات حسب مدة الخدمة والجنس",
        color_discrete_map=color_scheme
    )
    
    fig.update_layout(
        xaxis_title="مدة الخدمة",
        yaxis_title="عدد الاستقالات",
        height=400
    )
    
    return fig

def create_salary_analysis(data, color_scheme):
    """Create salary analysis charts with consistent colors"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary distribution by gender
        fig_salary_gender = px.box(
            data,
            x='الجنس',
            y='الراتب_الشهري',
            title="توزيع الرواتب حسب الجنس",
            color='الجنس',
            color_discrete_map=color_scheme
        )
        st.plotly_chart(fig_salary_gender, use_container_width=True)
    
    with col2:
        # Salary vs reasons
        salary_reason = data.groupby(['فئة_الراتب', 'السبب']).size().reset_index(name='العدد')
        top_reasons = data['السبب'].value_counts().head(5).index
        salary_reason_filtered = salary_reason[salary_reason['السبب'].isin(top_reasons)]
        
        fig_salary_reason = px.bar(
            salary_reason_filtered,
            x='فئة_الراتب',
            y='العدد',
            color='السبب',
            title="أسباب الترك حسب فئة الراتب"
        )
        fig_salary_reason.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_salary_reason, use_container_width=True)

def create_gender_comparison_line(data, color_scheme):
    """Create a comparison line chart showing monthly trends by gender"""
    # Create monthly data for demonstration
    months = ['يناير', 'فبراير', 'مارس', 'أبريل', 'مايو', 'يونيو',
             'يوليو', 'أغسطس', 'سبتمبر', 'أكتوبر', 'نوفمبر', 'ديسمبر']
    
    # Generate realistic monthly data
    np.random.seed(42)
    male_monthly = np.random.poisson(15, 12) + np.random.normal(0, 2, 12)
    female_monthly = np.random.poisson(10, 12) + np.random.normal(0, 1.5, 12)
    
    fig = go.Figure()
    
    # Add male line
    fig.add_trace(go.Scatter(
        x=months,
        y=male_monthly,
        mode='lines+markers',
        name='ذكر',
        line=dict(color=color_scheme['ذكر'], width=3),
        marker=dict(size=8, color=color_scheme['ذكر']),
        fill='tonexty'
    ))
    
    # Add female line
    fig.add_trace(go.Scatter(
        x=months,
        y=female_monthly,
        mode='lines+markers',
        name='أنثى',
        line=dict(color=color_scheme['أنثى'], width=3),
        marker=dict(size=8, color=color_scheme['أنثى']),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="النمط الشهري للاستقالات حسب الجنس",
        xaxis_title="الشهر",
        yaxis_title="عدد الاستقالات",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏢 نظام تحليل أسباب ترك الموظفين للعمل</h1>
        <h2>Employee Attrition Analysis System</h2>
        <p>تحليل شامل ومتقدم لأسباب واتجاهات ترك الموظفين</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = EmployeeAttritionAnalyzer()
    
    # Load and process data
    with st.spinner("جاري تحميل ومعالجة البيانات..."):
        sample_data = analyzer.load_sample_data()
        analyzer.process_data(sample_data)
    
    st.success("✅ تم تحميل ومعالجة البيانات بنجاح!")
    
    # Get summary statistics
    stats = analyzer.get_summary_stats()
    
    # Summary metrics
    st.subheader("📊 الملخص التنفيذي")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "إجمالي الحالات",
            f"{stats['total']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            "الموظفين الذكور",
            f"{stats['male_count']:,}",
            delta=f"{stats['male_pct']:.1f}%"
        )
    
    with col3:
        st.metric(
            "الموظفات الإناث",
            f"{stats['female_count']:,}",
            delta=f"{stats['female_pct']:.1f}%"
        )
    
    with col4:
        st.metric(
            "متوسط العمر",
            f"{stats['avg_age']:.1f} سنة",
            delta=None
        )
    
    with col5:
        st.metric(
            "متوسط مدة الخدمة",
            f"{stats['avg_service']:.1f} سنة",
            delta=None
        )
    
    # Insights section
    st.subheader("🔍 الرؤى الذكية")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for i, insight in enumerate(analyzer.insights[:len(analyzer.insights)//2]):
            if "🚨" in insight or "⚠️" in insight:
                st.markdown(f'<div class="warning-card">{insight}</div>', unsafe_allow_html=True)
            elif "📈" in insight or "📉" in insight:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-card">{insight}</div>', unsafe_allow_html=True)
    
    with col2:
        for insight in analyzer.insights[len(analyzer.insights)//2:]:
            if "🚨" in insight or "⚠️" in insight:
                st.markdown(f'<div class="warning-card">{insight}</div>', unsafe_allow_html=True)
            elif "📈" in insight or "📉" in insight:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-card">{insight}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main analysis sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 تحليل الأسباب", "👥 تحليل الجنس", "📈 الاتجاهات الزمنية", 
        "📉 المخططات الخطية", "🏢 تحليل الإدارات", "⏰ تحليل مدة الخدمة", "💰 تحليل الرواتب"
    ])
    
    with tab1:
        st.subheader("تحليل أسباب ترك العمل")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Reasons bar chart
            fig_reasons = create_reason_analysis_chart(analyzer.processed_data)
            st.plotly_chart(fig_reasons, use_container_width=True)
        
        with col2:
            # Reasons pie chart
            reason_counts = analyzer.processed_data['السبب'].value_counts().head(8)
            fig_pie = px.pie(
                values=reason_counts.values,
                names=reason_counts.index,
                title="توزيع أسباب الترك (أهم 8 أسباب)"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Gender-reason heatmap
        st.subheader("العلاقة بين الجنس وأسباب الترك")
        fig_heatmap = create_gender_reason_heatmap(analyzer.processed_data)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        st.subheader("تحليل الاستقالات حسب الجنس")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution pie chart
            gender_counts = analyzer.processed_data['الجنس'].value_counts()
            fig_gender_pie = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="توزيع الاستقالات حسب الجنس",
                color_discrete_map=analyzer.color_scheme
            )
            fig_gender_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_gender_pie, use_container_width=True)
        
        with col2:
            # Age distribution by gender
            fig_age_gender = px.histogram(
                analyzer.processed_data,
                x='السن',
                color='الجنس',
                title="توزيع الأعمار حسب الجنس",
                nbins=20,
                color_discrete_map=analyzer.color_scheme
            )
            st.plotly_chart(fig_age_gender, use_container_width=True)
        
        # Monthly comparison line chart
        st.subheader("مقارنة الاتجاهات الشهرية")
        fig_monthly = create_gender_comparison_line(analyzer.processed_data, analyzer.color_scheme)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab3:
        st.subheader("تحليل الاتجاهات الزمنية")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Yearly trend chart
            fig_yearly = create_yearly_trend_chart(analyzer.processed_data, analyzer.color_scheme)
            st.plotly_chart(fig_yearly, use_container_width=True)
        
        with col2:
            # Yearly total trend
            yearly_total = analyzer.processed_data.groupby('سنة_الترك').size().reset_index(name='العدد')
            fig_yearly_total = px.bar(
                yearly_total,
                x='سنة_الترك',
                y='العدد',
                title="إجمالي الاستقالات السنوية",
                color='العدد',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_yearly_total, use_container_width=True)
        
        # Detailed yearly analysis
        st.subheader("تحليل مفصل للاتجاهات السنوية")
        yearly_detailed = analyzer.processed_data.groupby(['سنة_الترك', 'الجنس', 'الإدارة']).size().reset_index(name='العدد')
        
        # Select department for detailed view
        selected_dept = st.selectbox("اختر الإدارة للتحليل المفصل:", 
                                   analyzer.processed_data['الإدارة'].unique())
        
        dept_data = yearly_detailed[yearly_detailed['الإدارة'] == selected_dept]
        fig_dept_yearly = px.line(
            dept_data,
            x='سنة_الترك',
            y='العدد',
            color='الجنس',
            title=f"اتجاه الاستقالات في إدارة {selected_dept}",
            markers=True,
            color_discrete_map=analyzer.color_scheme
        )
        st.plotly_chart(fig_dept_yearly, use_container_width=True)
    
    with tab4:
        st.subheader("المخططات الخطية المتقدمة")
        
        # Enhanced line graph
        fig_line = create_attrition_line_graph(analyzer.processed_data, analyzer.color_scheme)
        st.plotly_chart(fig_line, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Service duration trend
            service_yearly = analyzer.processed_data.groupby(['سنة_الترك', 'فئة_مدة_الخدمة']).size().reset_index(name='العدد')
            fig_service_trend = px.line(
                service_yearly,
                x='سنة_الترك',
                y='العدد',
                color='فئة_مدة_الخدمة',
                title="اتجاه الاستقالات حسب مدة الخدمة",
                markers=True
            )
            st.plotly_chart(fig_service_trend, use_container_width=True)
        
        with col2:
            # Age group trend
            age_yearly = analyzer.processed_data.groupby(['سنة_الترك', 'فئة_العمر']).size().reset_index(name='العدد')
            fig_age_trend = px.line(
                age_yearly,
                x='سنة_الترك',
                y='العدد',
                color='فئة_العمر',
                title="اتجاه الاستقالات حسب الفئة العمرية",
                markers=True
            )
            st.plotly_chart(fig_age_trend, use_container_width=True)
    
    with tab5:
        st.subheader("تحليل الإدارات والأقسام")
        
        # Department analysis
        fig_dept = create_department_analysis(analyzer.processed_data, analyzer.color_scheme)
        st.plotly_chart(fig_dept, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top departments by attrition
            dept_counts = analyzer.processed_data['الإدارة'].value_counts().head(10)
            fig_dept_bar = px.bar(
                x=dept_counts.values,
                y=dept_counts.index,
                orientation='h',
                title="أكثر الإدارات من حيث الاستقالات",
                labels={'x': 'عدد الاستقالات', 'y': 'الإدارة'}
            )
            st.plotly_chart(fig_dept_bar, use_container_width=True)
        
        with col2:
            # Department reasons analysis
            selected_dept_reason = st.selectbox("اختر الإدارة لتحليل الأسباب:", 
                                              analyzer.processed_data['الإدارة'].unique(),
                                              key="dept_reason_select")
            
            dept_reason_data = analyzer.processed_data[analyzer.processed_data['الإدارة'] == selected_dept_reason]
            reason_counts_dept = dept_reason_data['السبب'].value_counts().head(5)
            
            fig_dept_reasons = px.pie(
                values=reason_counts_dept.values,
                names=reason_counts_dept.index,
                title=f"أسباب الترك في إدارة {selected_dept_reason}"
            )
            st.plotly_chart(fig_dept_reasons, use_container_width=True)
    
    with tab6:
        st.subheader("تحليل مدة الخدمة")
        
        # Service duration analysis
        fig_service = create_service_duration_analysis(analyzer.processed_data, analyzer.color_scheme)
        st.plotly_chart(fig_service, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average service by department
            avg_service_dept = analyzer.processed_data.groupby('الإدارة')['مدة_الخدمة_بالسنوات'].mean().sort_values(ascending=False)
            fig_avg_service = px.bar(
                x=avg_service_dept.values,
                y=avg_service_dept.index,
                orientation='h',
                title="متوسط مدة الخدمة حسب الإدارة",
                labels={'x': 'متوسط مدة الخدمة (سنوات)', 'y': 'الإدارة'}
            )
            st.plotly_chart(fig_avg_service, use_container_width=True)
        
        with col2:
            # Service duration distribution
            fig_service_dist = px.histogram(
                analyzer.processed_data,
                x='مدة_الخدمة_بالسنوات',
                title="توزيع مدة الخدمة",
                nbins=20,
                labels={'x': 'مدة الخدمة (سنوات)', 'y': 'العدد'}
            )
            st.plotly_chart(fig_service_dist, use_container_width=True)
        
        # Service duration vs reasons
        st.subheader("العلاقة بين مدة الخدمة وأسباب الترك")
        service_reason = analyzer.processed_data.groupby(['فئة_مدة_الخدمة', 'السبب']).size().reset_index(name='العدد')
        top_reasons_service = analyzer.processed_data['السبب'].value_counts().head(5).index
        service_reason_filtered = service_reason[service_reason['السبب'].isin(top_reasons_service)]
        
        fig_service_reason = px.bar(
            service_reason_filtered,
            x='فئة_مدة_الخدمة',
            y='العدد',
            color='السبب',
            title="أسباب الترك حسب فئة مدة الخدمة"
        )
        st.plotly_chart(fig_service_reason, use_container_width=True)
    
    with tab7:
        st.subheader("تحليل الرواتب والتعويضات")
        
        # Salary analysis
        create_salary_analysis(analyzer.processed_data, analyzer.color_scheme)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average salary by department
            avg_salary_dept = analyzer.processed_data.groupby('الإدارة')['الراتب_الشهري'].mean().sort_values(ascending=False)
            fig_salary_dept = px.bar(
                x=avg_salary_dept.values,
                y=avg_salary_dept.index,
                orientation='h',
                title="متوسط الراتب حسب الإدارة",
                labels={'x': 'متوسط الراتب الشهري', 'y': 'الإدارة'}
            )
            st.plotly_chart(fig_salary_dept, use_container_width=True)
        
        with col2:
            # Salary vs service duration
            fig_salary_service = px.scatter(
                analyzer.processed_data,
                x='مدة_الخدمة_بالسنوات',
                y='الراتب_الشهري',
                color='الجنس',
                title="العلاقة بين الراتب ومدة الخدمة",
                color_discrete_map=analyzer.color_scheme
            )
            st.plotly_chart(fig_salary_service, use_container_width=True)
    
    # Recommendations section
    st.markdown("---")
    st.subheader("📋 التوصيات والإجراءات المقترحة")
    
    recommendations = analyzer.generate_recommendations()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="recommendations-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 توصيات فورية")
        for i, rec in enumerate(recommendations[:len(recommendations)//2]):
            st.markdown(f"**{i+1}.** {rec}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="recommendations-card">', unsafe_allow_html=True)
        st.markdown("### 📈 توصيات استراتيجية")
        for i, rec in enumerate(recommendations[len(recommendations)//2:], len(recommendations)//2+1):
            st.markdown(f"**{i}.** {rec}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional insights
    st.markdown("---")
    st.subheader("📈 تحليلات إضافية")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Risk indicators
        st.markdown("### ⚠️ مؤشرات الخطر")
        high_risk_depts = analyzer.processed_data['الإدارة'].value_counts().head(3)
        for dept, count in high_risk_depts.items():
            st.warning(f"**{dept}**: {count} استقالة")
    
    with col2:
        # Positive indicators
        st.markdown("### ✅ مؤشرات إيجابية")
        long_service = (analyzer.processed_data['مدة_الخدمة_بالسنوات'] > 5).sum()
        st.success(f"**{long_service}** موظف خدم أكثر من 5 سنوات")
        
        high_salary = (analyzer.processed_data['الراتب_الشهري'] > 12000).sum()
        st.success(f"**{high_salary}** موظف براتب عالي")
    
    with col3:
        # Key metrics
        st.markdown("### 📊 مقاييس أساسية")
        avg_age = analyzer.processed_data['السن'].mean()
        st.info(f"**متوسط عمر المستقيلين**: {avg_age:.1f} سنة")
        
        retention_rate = 100 - (len(analyzer.processed_data) / (len(analyzer.processed_data) + 5000) * 100)
        st.info(f"**معدل الاحتفاظ المقدر**: {retention_rate:.1f}%")
    
    # Export functionality
    st.markdown("---")
    st.subheader("📤 تصدير البيانات والتقارير")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("تصدير البيانات الخام", use_container_width=True):
            csv_data = analyzer.processed_data.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="تحميل CSV",
                data=csv_data,
                file_name="employee_attrition_data.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("تصدير الإحصائيات", use_container_width=True):
            stats_df = pd.DataFrame([stats])
            stats_csv = stats_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="تحميل الإحصائيات",
                data=stats_csv,
                file_name="attrition_statistics.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("تصدير التوصيات", use_container_width=True):
            recommendations_text = "\n".join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)])
            st.download_button(
                label="تحميل التوصيات",
                data=recommendations_text,
                file_name="recommendations.txt",
                mime="text/plain"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏢 نظام تحليل ترك الموظفين - تم تطويره باستخدام Streamlit و Plotly</p>
        <p>© 2024 - جميع الحقوق محفوظة</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()