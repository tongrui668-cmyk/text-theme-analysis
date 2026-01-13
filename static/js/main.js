// 错误处理工具函数
function handleError(message, error, element = null) {
    console.error(`${message}:`, error);
    if (element && element.parentElement) {
        element.parentElement.innerHTML = `<p class="error-message">${message}，请刷新页面重试</p>`;
    }
}

// 数据验证工具函数
function validateData(data, requiredKeys = []) {
    if (!data || typeof data !== 'object') {
        return false;
    }
    return requiredKeys.every(key => key in data);
}

// 返回顶部按钮
const backToTopButton = document.getElementById('backToTop');
if (backToTopButton) {
    window.addEventListener('scroll', () => {
        try {
            if (window.pageYOffset > 300) {
                backToTopButton.classList.add('show');
            } else {
                backToTopButton.classList.remove('show');
            }
        } catch (error) {
            console.error('滚动事件处理失败:', error);
        }
    });
    
    backToTopButton.addEventListener('click', () => {
        try {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        } catch (error) {
            console.error('返回顶部失败:', error);
        }
    });
}

// 示例评论展开/收起
function toggleExamples(theme, event) {
    try {
        if (!event || !event.target) {
            console.error('事件对象无效');
            return;
        }
        
        const hiddenDiv = document.getElementById(`examples-${theme}-hidden`);
        const button = event.target;
        
        if (!hiddenDiv) {
            console.error(`未找到元素: examples-${theme}-hidden`);
            return;
        }
        
        if (hiddenDiv.style.display === 'none' || hiddenDiv.style.display === '') {
            hiddenDiv.style.display = 'block';
            button.textContent = '收起示例';
        } else {
            hiddenDiv.style.display = 'none';
            try {
                const exampleCount = document.querySelectorAll(`#examples-${theme}-hidden .example-item`).length + 1;
                button.textContent = `显示全部 ${exampleCount} 条示例`;
            } catch (countError) {
                console.error('计算示例数量失败:', countError);
                button.textContent = '显示全部示例';
            }
        }
    } catch (error) {
        console.error('切换示例显示失败:', error);
    }
}

// 图表配置
function initCharts() {
    // 检查Chart.js是否加载
    if (typeof Chart === 'undefined') {
        console.error('Chart.js库未加载');
        return;
    }
    
    // 获取主题分析数据，优先使用实际数据，否则使用示例数据
    const analysisData = window.themeAnalysisData || {};
    const themeNames = analysisData.theme_names || { "0": "产品功能", "1": "用户体验", "2": "性能问题", "3": "价格策略", "4": "客户服务" };
    const themeData = analysisData.themes_count || { "0": 150, "1": 120, "2": 80, "3": 60, "4": 40 };
    const hasTimeData = analysisData.has_time_data || false;
    const timeSeriesData = analysisData.time_series_data || {
        "2023-01": { "0": 20, "1": 15, "2": 10 },
        "2023-02": { "0": 25, "1": 18, "2": 12 },
        "2023-03": { "0": 30, "1": 22, "2": 8 },
        "2023-04": { "0": 35, "1": 25, "2": 6 }
    };
    const keywordTimeSeries = analysisData.keyword_time_series || {
        "2023-01": { "功能": 15, "体验": 12, "速度": 8, "价格": 6, "服务": 4 },
        "2023-02": { "功能": 18, "体验": 15, "速度": 10, "价格": 7, "服务": 5 },
        "2023-03": { "功能": 22, "体验": 18, "速度": 7, "价格": 9, "服务": 6 },
        "2023-04": { "功能": 25, "体验": 20, "速度": 5, "价格": 10, "服务": 8 }
    };
    const topKeywords = analysisData.top_keywords || ["功能", "体验", "速度", "价格", "服务"];
    const themeConfidence = analysisData.theme_confidence || { "0": 0.85, "1": 0.82, "2": 0.78, "3": 0.75, "4": 0.80 };
    
    // 主题分布饼图
    const pieCtx = document.getElementById('themePieChart');
    if (pieCtx) {
        try {
            // 验证数据
            if (!validateData(themeData) || Object.keys(themeData).length === 0) {
                throw new Error('主题数据无效');
            }
            
            const labels = Object.keys(themeData).map(theme => themeNames[theme] || theme);
            const chartData = Object.values(themeData);
            
            // 生成柔和的颜色
            const backgroundColors = [
                '#5470C6', // 天蓝
                '#91CC75', // 薄荷绿
                '#FAC858', // 柠檬黄
                '#EE6666', // 草莓红
                '#73C0DE', // 浅青
                '#BA72D2', // 薰衣草紫
                '#F98686'  // 西柚粉
            ];
            
            new Chart(pieCtx, {
                type: 'pie',
                data: {
                    labels: labels,
                    datasets: [{
                        data: chartData,
                        backgroundColor: backgroundColors.slice(0, labels.length),
                        borderColor: backgroundColors.slice(0, labels.length),
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                font: {
                                    size: 12
                                },
                                padding: 20
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    try {
                                        const label = context.label || '';
                                        const value = context.raw || 0;
                                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                        const percentage = Math.round((value / total) * 100);
                                        return `${label}: ${value} (${percentage}%)`;
                                    } catch (error) {
                                        console.error('饼图tooltip回调失败:', error);
                                        return context.label || '';
                                    }
                                }
                            }
                        }
                    }
                }
            });
        } catch (error) {
            handleError('无法加载饼图', error, pieCtx);
        }
    }
    
    // 主题分布条形图
    const barCtx = document.getElementById('themeBarChart');
    if (barCtx) {
        try {
            // 验证数据
            if (!validateData(themeData) || Object.keys(themeData).length === 0) {
                throw new Error('主题数据无效');
            }
            
            const labels = Object.keys(themeData).map(theme => themeNames[theme] || theme);
            const chartData = Object.values(themeData);
            
            new Chart(barCtx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '评论数量',
                        data: chartData,
                        backgroundColor: '#5470C6',
                        borderColor: '#3949AB',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    try {
                                        const value = context.raw || 0;
                                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                        const percentage = Math.round((value / total) * 100);
                                        return `数量: ${value} (${percentage}%)`;
                                    } catch (error) {
                                        console.error('条形图tooltip回调失败:', error);
                                        return `数量: ${context.raw || 0}`;
                                    }
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            });
        } catch (error) {
            handleError('无法加载条形图', error, barCtx);
        }
    }
    
    // 主题趋势图
    const trendCtx = document.getElementById('themeTrendChart');
    if (trendCtx) {
        try {
            // 验证数据
            if (!validateData(timeSeriesData) || Object.keys(timeSeriesData).length === 0) {
                throw new Error('时间序列数据无效');
            }
            
            // 处理时间序列数据
            const labels = Object.keys(timeSeriesData);
            const datasets = [];
            const colors = [
                '#5470C6', // 天蓝
                '#91CC75', // 薄荷绿
                '#FAC858', // 柠檬黄
                '#EE6666', // 草莓红
                '#73C0DE', // 浅青
                '#BA72D2', // 薰衣草紫
                '#F98686'  // 西柚粉
            ];
            
            // 获取所有主题
            const allThemes = new Set();
            Object.values(timeSeriesData).forEach(dataPoint => {
                if (dataPoint && typeof dataPoint === 'object') {
                    Object.keys(dataPoint).forEach(theme => allThemes.add(theme));
                }
            });
            
            Array.from(allThemes).forEach((theme, index) => {
                const themeData = labels.map(label => {
                    const timePointData = timeSeriesData[label];
                    return timePointData && timePointData[theme] ? timePointData[theme] : 0;
                });
                datasets.push({
                    label: themeNames[theme] || theme,
                    data: themeData,
                    borderColor: colors[index % colors.length],
                    backgroundColor: colors[index % colors.length] + '80',
                    tension: 0.3,
                    fill: false
                });
            });
            
            new Chart(trendCtx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                font: {
                                    size: 12
                                },
                                padding: 20
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            });
        } catch (error) {
            handleError('无法加载趋势图', error, trendCtx);
        }
    }
    
    // 关键词趋势图
    const keywordTrendCtx = document.getElementById('keywordTrendChart');
    if (keywordTrendCtx) {
        try {
            // 验证数据
            if (!validateData(keywordTimeSeries) || Object.keys(keywordTimeSeries).length === 0) {
                throw new Error('关键词时间序列数据无效');
            }
            
            if (!Array.isArray(topKeywords) || topKeywords.length === 0) {
                throw new Error('顶部关键词数据无效');
            }
            
            // 处理关键词时间序列数据
            const labels = Object.keys(keywordTimeSeries);
            const datasets = [];
            const colors = [
                '#5470C6', // 天蓝
                '#91CC75', // 薄荷绿
                '#FAC858', // 柠檬黄
                '#EE6666', // 草莓红
                '#73C0DE', // 浅青
                '#BA72D2', // 薰衣草紫
                '#F98686'  // 西柚粉
            ];
            
            topKeywords.slice(0, 5).forEach((keyword, index) => {
                const keywordData = labels.map(label => {
                    const timePointData = keywordTimeSeries[label];
                    return timePointData && timePointData[keyword] ? timePointData[keyword] : 0;
                });
                datasets.push({
                    label: keyword,
                    data: keywordData,
                    borderColor: colors[index % colors.length],
                    backgroundColor: colors[index % colors.length] + '80',
                    tension: 0.3,
                    fill: false
                });
            });
            
            new Chart(keywordTrendCtx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                font: {
                                    size: 12
                                },
                                padding: 20
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            });
        } catch (error) {
            handleError('无法加载关键词趋势图', error, keywordTrendCtx);
        }
    }
    
    // 主题对比图
    const comparisonCtx = document.getElementById('themeComparisonChart');
    if (comparisonCtx) {
        try {
            // 验证数据
            if (!validateData(themeData) || Object.keys(themeData).length === 0) {
                throw new Error('主题数据无效');
            }
            
            const labels = Object.keys(themeData).map(theme => themeNames[theme] || theme);
            const chartData = Object.values(themeData);
            
            new Chart(comparisonCtx, {
                type: 'radar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '评论数量',
                        data: chartData,
                        backgroundColor: '#5470C620',
                        borderColor: '#5470C6',
                        pointBackgroundColor: '#5470C6',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#5470C6'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            angleLines: {
                                display: true,
                                color: 'rgba(0, 0, 0, 0.05)'
                            },
                            suggestedMin: 0
                        }
                    }
                }
            });
        } catch (error) {
            handleError('无法加载主题对比图', error, comparisonCtx);
        }
    }
    
    // 置信度对比图
    const confidenceCtx = document.getElementById('confidenceComparisonChart');
    if (confidenceCtx) {
        try {
            // 验证数据
            if (!validateData(themeConfidence) || Object.keys(themeConfidence).length === 0) {
                throw new Error('主题置信度数据无效');
            }
            
            const labels = Object.keys(themeConfidence).map(theme => themeNames[theme] || theme);
            const chartData = Object.values(themeConfidence);
            
            new Chart(confidenceCtx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '平均置信度',
                        data: chartData,
                        backgroundColor: '#5470C6',
                        borderColor: '#3949AB',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            });
        } catch (error) {
            handleError('无法加载置信度对比图', error, confidenceCtx);
        }
    }
}

// 当DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
});