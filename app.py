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

            forecast_result = forecast[["ds", "yhat", "delta", "pct_change"]].tail(forecast_months)

            # Tạo nhận xét tổng quan
            forecasted_mean = forecast_result["yhat"].mean()
            pct_total_change = (forecasted_mean - recent_avg) / recent_avg * 100
            if pct_total_change > 10:
                comment = f"📈 Doanh thu dự kiến TĂNG khoảng {pct_total_change:.1f}% so với trung bình 3 tháng gần nhất."
            elif pct_total_change < -10:
                comment = f"📉 Doanh thu dự kiến GIẢM khoảng {abs(pct_total_change):.1f}% so với trung bình 3 tháng gần nhất."
            else:
                comment = "➖ Doanh thu dự kiến ỔN ĐỊNH, không biến động lớn."

            # Hiển thị bảng kết quả
            st.subheader("📊 Kết quả Dự báo")
            def highlight_delta(val):
                return "color: red;" if abs(val) > threshold else ""

            st.dataframe(
                forecast_result.style.format({"yhat": "{:.2f}", "delta": "{:.2f}", "pct_change": "{:.1f}%"})
                .applymap(highlight_delta, subset=["pct_change"])
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
                date = row["ds"]
                yhat = row["yhat"]
                delta = row["delta"]
                pct = row["pct_change"]
                month_label = f"Tháng {date.month}/{date.year}"

                if yhat > recent_avg:
                    suggestions.append(f"🟢 {month_label}: Xu hướng TĂNG. Xem xét tăng nhập hàng và tối ưu giá bán.")
                else:
                    suggestions.append(f"🔵 {month_label}: Xu hướng GIẢM. Cần cân nhắc khuyến mãi hoặc giảm hàng tồn.")

                if abs(pct) > threshold:
                    suggestions.append(f"⚠️ {month_label}: Biến động doanh thu {pct:.1f}%. Rủi ro tồn kho.")

            suggestions.append("💡 Duy trì theo dõi định kỳ. Cập nhật mô hình mỗi tháng để phản ánh biến động mới.")
            for s in suggestions:
                st.write(s)
