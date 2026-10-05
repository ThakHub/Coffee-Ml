import streamlit as st

st.set_page_config(
    page_title="Coffee Bean Classification",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Application Coffee Bean Classification")
st.header("กลุ่ม Learning")
st.subheader(":blue[Applied Machine Learning 2/69]")
st.subheader(":red[จัดทำโดย]")
st.subheader(":orange[นายทักษ์ดนัย] :green[วงษ์คำนา]")
st.subheader(":orange[นางสาวภัทรฤดี] :green[นราวิริยะกุล]")
st.subheader(":orange[นายธนวรรธน์] :green[สละทาน]")

st.divider()

st.markdown("""
### เกี่ยวกับโปรเจกต์นี้
เว็บแอปพลิเคชันนี้ใช้สำหรับ **จำแนกสายพันธุ์เมล็ดกาแฟ** จากภาพถ่าย โดยใช้โมเดล Deep Learning (Transfer Learning)

**สายพันธุ์ที่รองรับ**
- 🟢 **Arabica**
- 🟡 **Excelsa**
- 🟤 **Liberica**
- 🔴 **Robusta**

**โมเดลที่ใช้**
- DenseNet121 (Transfer Learning จาก ImageNet)
- ความถูกต้องบนชุดทดสอบ: **70.81%**

---
ไปที่เมนู **Coffee Classification** เพื่อทดลองอัปโหลดภาพและทำนายผล
""")