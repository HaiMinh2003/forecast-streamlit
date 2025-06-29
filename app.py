import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dự báo Doanh thu Sản phẩm", layout="wide")
st.title("🔮 Dự báo Doanh thu Sản phẩm theo Tháng")

# Tải file CSV
uploaded_file = st.file_uploader("📂 Chọn file CSV dữ liệu", type=["csv"])

if uploaded_file:
    # Đọc dữ liệu CSV
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df.dropna(subset=["InvoiceDate", "StockCode", "Quantity", "UnitPrice"], inplace=True)
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Lấy danh sách sản phẩm và quốc gia
    stock_codes = sorted(df["StockCode"].unique())
    countries = sorted(df["Country"].unique())

    # Giao diện nhập tham số
    col1, col2 = st.columns(2)
    stock_code = col1.selectbox("🛒 Chọn sản phẩm", stock_codes)
    country = col2.selectbox("🌎 Chọn quốc gia", countries)
    forecast_months = st.number_input("📆 Số tháng cần dự báo", min_value=1, value=3, step=1)
    threshold = st.number_input("⚠️ Ngưỡng cảnh báo (%)", min_value=0.0, value=10.0, step=1.0)

    if st.button("🚀 Chạy dự báo"):
        # Lọc dữ liệu
        filtered_df = df[(df["StockCode"] == stock_code) & (df["Country"] == country)]
        if filtered_df.empty:
            st.error("❌ Không có dữ liệu phù hợp.")
        else:
            monthly = (
                filtered_df.groupby("Month").agg({"Revenue": "sum"}).reset_index()
            )
            monthly.columns = ["ds", "y"]

            # Mô hình Prophet
            model = Prophet()
            model.fit(monthly)

            future = model.make_future_dataframe(periods=forecast_months, freq="MS")
            forecast = model.predict(future)

            # Tính trung bình 3 tháng gần nhất
            recent_avg = monthly["y"].tail(3).mean()
            forecast["delta"] = forecast["yhat"] - recent_avg
            forecast["pct_change"] = 100 * forecast["delta"] / recent_avg

            # Tạo dataframe kết quả KHÔNG CÓ CỘT STT
            forecast_result_raw = forecast[["ds", "yhat", "delta", "pct_change"]].tail(forecast_months)
            forecast_result = pd.DataFrame({
                "Tháng dự báo": forecast_result_raw["ds"].dt.strftime("%m/%Y"),
                "Doanh thu dự báo": forecast_result_raw["yhat"],
                "Chênh lệch": forecast_result_raw["delta"],
                "So với TB 3T (%)": forecast_result_raw["pct_change"]
            })

            # Tạo nhận xét tổng quan chi tiết
            forecasted_mean = forecast_result["Doanh thu dự báo"].mean()
            forecasted_min = forecast_result["Doanh thu dự báo"].min()
            forecasted_max = forecast_result["Doanh thu dự báo"].max()
            pct_total_change = (forecasted_mean - recent_avg) / recent_avg * 100

            if pct_total_change > 10:
                trend_desc = "xu hướng TĂNG rõ rệt"
            elif pct_total_change < -10:
                trend_desc = "xu hướng GIẢM đáng kể"
            else:
                trend_desc = "xu hướng ỔN ĐỊNH"

            comment = (
                f"🔍 Trong {forecast_months} tháng dự báo, "
                f"doanh thu trung bình dự kiến đạt {forecasted_mean:.1f}, "
                f"{'tăng' if pct_total_change >=0 else 'giảm'} {abs(pct_total_change):.1f}% so với trung bình 3 tháng gần nhất.\n\n"
                f"Doanh thu dự báo dao động từ {forecasted_min:.1f} đến {forecasted_max:.1f}, "
                f"thể hiện {trend_desc}."
            )

            # Hiển thị bảng kết quả
            st.subheader("📊 Kết quả Dự báo")
            def highlight_delta(val):
                return "color: red;" if abs(val) > threshold else ""

            st.dataframe(
                forecast_result.style.format({
                    "Doanh thu dự báo": "{:.2f}",
                    "Chênh lệch": "{:.2f}",
                    "So với TB 3T (%)": "{:.1f}%"
                }).applymap(
                    lambda v: "color: red;" if isinstance(v, (float, int)) and abs(v) > threshold else "",
                    subset=["So với TB 3T (%)"]
                )
            )

            # Hiển thị biểu đồ
            st.subheader("📈 Biểu đồ Dự báo")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(forecast["ds"], forecast["yhat"], label="Dự báo")
            ax.set_xlabel("Thời gian")
            ax.set_ylabel("Doanh thu dự báo")
            ax.set_title("Diễn biến doanh thu dự báo")
            ax.legend()
            st.pyplot(fig)

            # Hiển thị nhận xét tổng quan
            st.info(comment)

            # Hiển thị gợi ý chi tiết
            st.subheader("🔍 Phân tích & Gợi ý")

            suggestions = []
            for _, row in forecast_result.iterrows():
                month_label = row["Tháng dự báo"]
                yhat = row["Doanh thu dự báo"]
                pct = row["So với TB 3T (%)"]

                # Phân loại xu hướng chi tiết
                if pct >= 10:
                    trend = "📈 Tăng mạnh"
                    action = "Tăng tồn kho 15–25% và đẩy mạnh quảng cáo."
                elif 5 <= pct < 10:
                    trend = "🟢 Tăng nhẹ"
                    action = "Tăng tồn kho 5–10% và duy trì hoạt động marketing."
                elif -5 < pct < 5:
                    trend = "➖ Ổn định"
                    action = "Duy trì tồn kho và chiến lược hiện tại."
                elif -10 < pct <= -5:
                    trend = "🔵 Giảm nhẹ"
                    action = "Cân nhắc giảm giá 5–10% hoặc triển khai khuyến mãi."
                else:
                    trend = "📉 Giảm mạnh"
                    action = "Giảm giá 10–20% và thanh lý hàng tồn."

                # Cảnh báo doanh thu thấp
                if yhat < 50:
                    low_revenue_note = "⚠️ Doanh thu rất thấp, cần xem xét điều chỉnh sản phẩm hoặc thị trường."
                else:
                    low_revenue_note = ""

                # Cảnh báo biến động vượt threshold
                if abs(pct) > threshold:
                    volatility_note = f"⚠️ Biến động {pct:.1f}% vượt ngưỡng cảnh báo."
                else:
                    volatility_note = ""

                suggestion_text = f"""
**{month_label}**
- Xu hướng: {trend}
- Đề xuất: {action}
{low_revenue_note}
{volatility_note}
"""
                suggestions.append(suggestion_text)

            # Gợi ý tổng
            suggestions.append("💡 Duy trì theo dõi định kỳ và cập nhật mô hình hàng tháng để phản ánh biến động mới.")

            for s in suggestions:
                st.markdown(s)
