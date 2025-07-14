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

            # Cập nhật mô tả xu hướng
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
            st.dataframe(
                forecast_result.style.format({
                    "Doanh thu dự báo": "{:.2f}",
                    "Chênh lệch": "{:.2f}",
                    "So với TB 3T (%)": "{:.1f}%"
                })
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

            # Hiển thị gợi ý chi tiết dựa trên các mốc cảnh báo
            st.subheader("🔍 Phân tích & Gợi ý")
            suggestions = []
            for _, row in forecast_result.iterrows():
                month_label = row["Tháng dự báo"]
                pct = row["So với TB 3T (%)"]

                # Dựa trên % thay đổi, đưa ra hành động và gợi ý chi tiết
                if pct >= 10:
                    trend = "📈 Tăng rất mạnh"
                    action = "Mở rộng sản xuất và tăng cường cung cấp sản phẩm.\nĐẩy mạnh các chiến dịch quảng bá và bán hàng."
                    detail = "Tăng cường quảng bá sản phẩm và mở rộng chiến dịch marketing.\nXem xét hợp tác với các KOL/KOC để mở rộng thị trường."
                elif 5 <= pct < 10:
                    trend = "🟢 Tăng mạnh"
                    action = "Tiếp tục duy trì chiến lược marketing hiện tại.\nXem xét mở rộng sản xuất và tăng cường cung cấp sản phẩm."
                    detail = "Tiếp tục duy trì các chiến lược marketing đang hoạt động hiệu quả.\nKhám phá các thị trường mới và đầu tư vào cải tiến sản phẩm."
                elif 0 <= pct < 5:
                    trend = "➖ Tăng nhẹ"
                    action = "Duy trì chiến lược marketing hiện tại.\nTăng cường quảng bá sản phẩm và khuyến mãi."
                    detail = "Xem xét các kênh quảng cáo hiệu quả hơn (ví dụ: TikTok, Facebook, Instagram).\nTăng cường hợp tác với các KOL/KOC."
                elif -5 < pct < 0:
                    trend = "🔵 Giảm nhẹ"
                    action = "Cải thiện chiến lược marketing để duy trì ổn định.\nXem xét các chiến lược khuyến mãi."
                    detail = "Điều chỉnh mức giá sản phẩm để cải thiện lợi nhuận.\nTập trung vào nâng cao trải nghiệm khách hàng."
                elif -10 < pct <= -5:
                    trend = "📉 Giảm mạnh"
                    action = "Cần thay đổi chiến lược marketing hoàn toàn để thu hút khách hàng mới.\nTăng cường các chương trình khuyến mãi mạnh mẽ."
                    detail = "Tổ chức các sự kiện bán hàng đặc biệt hoặc flash sale.\nTăng cường chiến dịch quảng cáo trực tuyến và giảm giá mạnh."
                else:
                    trend = "🚨 Giảm rất mạnh"
                    action = "Điều chỉnh ngay lập tức chiến lược marketing.\nGiảm giá mạnh và thanh lý hàng tồn kho."
                    detail = "Cân nhắc giảm giá 10–20% hoặc thanh lý hàng tồn kho.\nTổ chức chiến dịch quảng cáo mạnh mẽ hơn và tăng ngân sách truyền thông."

                # Hiển thị các gợi ý hành động
                suggestions.append(f"**{month_label}** - Xu hướng: {trend}\n- Đề xuất: {action}\n- Gợi ý chi tiết: {detail}\n")

            # Hiển thị các gợi ý hành động
            for suggestion in suggestions:
                st.markdown(suggestion)

            # Gợi ý tổng
            st.markdown("💡 Duy trì theo dõi định kỳ và cập nhật mô hình hàng tháng để phản ánh biến động mới.")
