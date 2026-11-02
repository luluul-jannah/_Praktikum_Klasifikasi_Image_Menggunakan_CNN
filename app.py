"""
Dashboard Streamlit - Klasifikasi Gambar Biota Laut (OceanicLife) dengan CNN
Model    : MobileNetV2 (transfer learning), input 160x160 RGB
Jalankan : streamlit run app.py

Letakkan file ini sejajar dengan folder `output_oceaniclife`.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ==========================================
# PENGATURAN
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output_oceaniclife")

MODEL_CANDIDATES = [
    os.path.join(OUT_DIR, "final_marine_cnn_model.keras"),
    os.path.join(OUT_DIR, "oceaniclife_cnn_best.keras"),
]

IMG_SIZE = (160, 160)  # harus sama dengan saat training

DEFAULT_CLASS_NAMES = [
    "Fish", "Coral", "Shark", "Octopuses",
    "Crustaceans", "Seaweed", "Sponges", "Other marine organisms",
]

st.set_page_config(
    page_title="Dashboard Klasifikasi Biota Laut",
    page_icon="🐟",
    layout="wide",
)

# ==========================================
# FUNGSI PEMBACA FILE HASIL EKSPERIMEN
# ==========================================


def path(name):
    return os.path.join(OUT_DIR, name)


@st.cache_data
def load_json(name):
    p = path(name)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_csv(name, **kwargs):
    p = path(name)
    if os.path.exists(p):
        return pd.read_csv(p, **kwargs)
    return None


@st.cache_data
def load_npy(name):
    p = path(name)
    if os.path.exists(p):
        return np.load(p)
    return None


@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    import tensorflow as tf

    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            return tf.keras.models.load_model(p), p
    return None, None


def show_figure_file(name, caption=None):
    p = path(name)
    if os.path.exists(p):
        st.image(p, caption=caption, use_container_width=True)
    else:
        st.info(f"Gambar `{name}` belum ada di folder `output_oceaniclife`.")


def analisis_box(text):
    """Kotak analisis di bawah setiap gambar/grafik."""
    if text:
        st.info(f"**Analisis:** {text}")


# ==========================================
# MEMUAT DATA
# ==========================================

metrics = load_json("metrics.json") or {}
history = load_json("history.json") or {}
eda = load_json("eda.json") or {}
analysis = load_json("analysis.json") or {}
class_names = (
    load_json("class_names.json") or metrics.get("class_names") or DEFAULT_CLASS_NAMES
)
per_class = load_csv("per_class.csv", index_col=0)
labels_df = load_csv("oceaniclife_labels.csv")
cm = load_npy("confusion_matrix.npy")

if labels_df is not None:
    labels_df = labels_df.dropna(subset=["label"]).copy()

# ==========================================
# JUDUL & DESKRIPSI
# ==========================================

st.title("🐟 Dashboard Klasifikasi Gambar Biota Laut")
st.markdown(
    """
Dashboard ini menampilkan hasil eksperimen **Convolutional Neural Network (CNN)** untuk
mengklasifikasikan gambar biota laut ke dalam **8 kelas** (dataset *OceanicLife*).
Model dibangun dengan *transfer learning* **MobileNetV2** (bobot ImageNet) dalam dua fase:
pelatihan *head* klasifikasi lalu *fine-tuning*. Seluruh gambar di-resize ke
**160×160 piksel** dan dibaca sebagai RGB.

Gunakan tab di bawah untuk melihat ringkasan dataset, distribusi kelas, hasil pelatihan,
evaluasi, confusion matrix, serta **uji prediksi gambar baru secara langsung**.
"""
)

with st.sidebar:
    st.header("Informasi Model")
    st.write("**Arsitektur:** MobileNetV2 + GAP + Dropout + Dense (softmax)")
    st.write(f"**Ukuran input:** {IMG_SIZE[0]}×{IMG_SIZE[1]}×3")
    st.write(f"**Jumlah kelas:** {len(class_names)}")
    if metrics:
        st.write(f"**Batch size:** {metrics.get('batch_size', '-')}")
        st.write(
            f"**Epoch:** fase 1 = {metrics.get('epochs_fase1', '-')}, "
            f"fase 2 = {metrics.get('epochs_fase2', '-')}"
        )
    st.markdown("**Daftar kelas:**")
    for i, name in enumerate(class_names):
        st.write(f"{i}. {name}")

tabs = st.tabs(
    [
        "📋 Ringkasan Dataset",
        "📊 Distribusi Data",
        "📈 Hasil Pelatihan",
        "🎯 Evaluasi Model",
        "🧩 Confusion Matrix",
        "🔍 Prediksi Gambar Baru",
    ]
)

# ==========================================
# TAB 1: RINGKASAN DATASET
# ==========================================

with tabs[0]:
    st.header("Ringkasan Dataset")

    n_train = metrics.get("n_train")
    n_val = metrics.get("n_val")
    n_test = metrics.get("n_test")
    n_labeled = len(labels_df) if labels_df is not None else None

    if n_labeled is None and None not in (n_train, n_val, n_test):
        n_labeled = n_train + n_val + n_test

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah kelas", len(class_names))
    c2.metric("Gambar berlabel", f"{n_labeled:,}" if n_labeled else "-")
    c3.metric("Data latih", f"{n_train:,}" if n_train else "-")
    c4.metric(
        "Data validasi / uji",
        f"{n_val:,} / {n_test:,}" if (n_val is not None and n_test is not None) else "-",
    )

    st.subheader("Proporsi data latih - validasi - uji")
    if None not in (n_train, n_val, n_test):
        split_df = pd.DataFrame(
            {"Subset": ["Latih", "Validasi", "Uji"], "Jumlah": [n_train, n_val, n_test]}
        )
        split_df["Proporsi (%)"] = split_df["Jumlah"] / split_df["Jumlah"].sum() * 100

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.dataframe(
                split_df.style.format({"Jumlah": "{:,}", "Proporsi (%)": "{:.1f}"}),
                hide_index=True,
                use_container_width=True,
            )
        with col_b:
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.pie(
                split_df["Jumlah"],
                labels=split_df["Subset"],
                autopct="%1.1f%%",
                startangle=90,
                colors=["#4C78A8", "#F58518", "#54A24B"],
            )
            ax.set_title("Proporsi Pembagian Data")
            st.pyplot(fig)
            plt.close(fig)

        tr_pct = n_train / (n_train + n_val + n_test) * 100
        analisis_box(
            f"Data dibagi menjadi {n_train:,} latih ({tr_pct:.0f}%), {n_val:,} validasi, "
            f"dan {n_test:,} uji. Data latih dipakai untuk memperbarui bobot, data validasi "
            "untuk memantau overfitting dan memilih model terbaik, sedangkan data uji "
            "disimpan terpisah agar evaluasi akhir tidak bias."
        )
    else:
        st.warning("Informasi pembagian data tidak ditemukan di `metrics.json`.")

    if eda:
        with st.expander("Detail EDA (eda.json)"):
            st.json(eda)

    st.subheader("Karakteristik gambar")
    show_figure_file("3_image_properties.png", "Properti gambar (ukuran & mode warna)")
    analisis_box(
        analysis.get("eda_ukuran")
        or "Ukuran dan mode warna gambar tidak seragam, sehingga semua gambar di-resize "
        "ke 160×160 dan dikonversi ke RGB."
    )

    st.subheader("Contoh gambar per kelas")
    show_figure_file("2_class_examples.png", "Contoh gambar tiap kelas")
    analisis_box(
        analysis.get("eda_contoh")
        or "Contoh visual menunjukkan variasi sudut pandang, pencahayaan, dan latar belakang "
        "antar gambar dalam satu kelas."
    )

    st.subheader("Sampel data")
    show_figure_file("4_sample_data.png", "Sampel data pelatihan setelah preprocessing")
    analisis_box(
        "Sampel ini menunjukkan gambar setelah di-resize dan dibaca sebagai RGB, yaitu "
        "bentuk yang sama dengan yang diterima model. Pemeriksaan visual ini memastikan "
        "label sesuai dengan isi gambar dan tidak ada gambar yang rusak."
    )

# ==========================================
# TAB 2: DISTRIBUSI DATA
# ==========================================

with tabs[1]:
    st.header("Distribusi Data per Kelas")

    counts = None
    if labels_df is not None and len(labels_df) > 0:
        counts = (
            labels_df["label"].value_counts().reindex(class_names, fill_value=0)
        )

    if counts is None:
        st.warning("File `oceaniclife_labels.csv` tidak ditemukan atau masih kosong.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        bars = ax.bar(counts.index, counts.values, color="steelblue", edgecolor="black")
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{int(b.get_height()):,}",
                ha="center",
                va="bottom",
            )
        ax.set_title("Distribusi Jumlah Gambar per Kelas")
        ax.set_xlabel("Kelas")
        ax.set_ylabel("Jumlah gambar")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        total = int(counts.sum())
        top_c, low_c = counts.idxmax(), counts.idxmin()
        ratio = counts.max() / max(counts.min(), 1)
        if ratio < 1.5:
            level = "relatif seimbang"
        elif ratio < 3:
            level = "cukup tidak seimbang"
        else:
            level = "sangat tidak seimbang"
        analisis_box(
            f"Dari {total:,} gambar berlabel, kelas terbanyak adalah '{top_c}' "
            f"({counts.max():,}; {counts.max() / total:.1%}) dan tersedikit '{low_c}' "
            f"({counts.min():,}; {counts.min() / total:.1%}). Rasio terbesar terhadap terkecil "
            f"{ratio:.1f}×, sehingga distribusi {level}. Pada data yang tidak seimbang, model "
            "cenderung lebih baik pada kelas mayoritas, jadi perhatikan recall dan F1 per kelas, "
            "bukan hanya akurasi."
        )

        table = pd.DataFrame(
            {"Jumlah": counts, "Persentase (%)": (counts / total * 100).round(2)}
        )
        st.dataframe(table, use_container_width=True)

    with st.expander("Gambar distribusi dari notebook"):
        show_figure_file("1_class_distribution.png", "Distribusi kelas (dari notebook)")

# ==========================================
# TAB 3: HASIL PELATIHAN
# ==========================================

with tabs[2]:
    st.header("Hasil Pelatihan (Akurasi & Loss)")

    if history and "loss" in history:
        epochs = range(1, len(history["loss"]) + 1)
        split = metrics.get("epochs_fase1")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

        axes[0].plot(epochs, history.get("accuracy", []), label="Train", marker="o", ms=3)
        axes[0].plot(epochs, history.get("val_accuracy", []), label="Validasi", marker="o", ms=3)
        axes[0].set_title("Akurasi")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")

        axes[1].plot(epochs, history["loss"], label="Train", marker="o", ms=3)
        axes[1].plot(epochs, history.get("val_loss", []), label="Validasi", marker="o", ms=3)
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")

        for ax in axes:
            if split:
                ax.axvline(split + 0.5, color="gray", linestyle="--", label="Mulai fine-tuning")
            ax.grid(alpha=0.4)
            ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        acc_tr = history["accuracy"][-1] if "accuracy" in history else None
        acc_va = history["val_accuracy"][-1] if "val_accuracy" in history else None
        c1, c2, c3, c4 = st.columns(4)
        if acc_tr is not None:
            c1.metric("Akurasi akhir (train)", f"{acc_tr:.2%}")
        if acc_va is not None:
            c2.metric("Akurasi akhir (validasi)", f"{acc_va:.2%}")
        c3.metric("Loss akhir (train)", f"{history['loss'][-1]:.4f}")
        if "val_loss" in history:
            c4.metric("Loss akhir (validasi)", f"{history['val_loss'][-1]:.4f}")

        analisis_box(
            analysis.get("kurva")
            or "Loss yang turun lalu mendatar menandakan model konvergen. Jarak kecil antara "
            "kurva train dan validasi menandakan model tidak overfitting berat."
        )
    else:
        st.warning("File `history.json` tidak ditemukan.")

    with st.expander("Gambar kurva pelatihan dari notebook"):
        show_figure_file("5_training_curves.png", "Kurva pelatihan (dari notebook)")

# ==========================================
# TAB 4: EVALUASI MODEL
# ==========================================

with tabs[3]:
    st.header("Hasil Evaluasi Model (Data Uji)")

    if metrics:
        macro = metrics.get("macro", {})
        weighted = metrics.get("weighted", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        c2.metric("Precision (macro)", f"{macro.get('precision', 0):.2%}")
        c3.metric("Recall (macro)", f"{macro.get('recall', 0):.2%}")
        c4.metric("F1-score (macro)", f"{macro.get('f1', 0):.2%}")

        summary = pd.DataFrame(
            {
                "Accuracy": [metrics.get("accuracy", 0)] * 2,
                "Precision": [macro.get("precision", 0), weighted.get("precision", 0)],
                "Recall": [macro.get("recall", 0), weighted.get("recall", 0)],
                "F1-score": [macro.get("f1", 0), weighted.get("f1", 0)],
            },
            index=["Macro avg", "Weighted avg"],
        )
        st.dataframe(summary.style.format("{:.4f}"), use_container_width=True)
        analisis_box(
            analysis.get("metrik")
            or "Macro F1 menunjukkan keseimbangan precision dan recall antar kelas; selisihnya "
            "dengan Weighted F1 menunjukkan pengaruh kelas mayoritas."
        )
    else:
        st.warning("File `metrics.json` tidak ditemukan.")

    st.subheader("Metrik per kelas")
    if per_class is not None:
        st.dataframe(
            per_class.style.format(
                {"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}
            ),
            use_container_width=True,
        )

        metric_choice = st.radio(
            "Tampilkan grafik untuk:", ["f1-score", "precision", "recall"], horizontal=True
        )
        fig, ax = plt.subplots(figsize=(10, 4.8))
        vals = per_class[metric_choice]
        bars = ax.bar(vals.index, vals.values, color="#54A24B", edgecolor="black")
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():.2f}", ha="center", va="bottom",
            )
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(metric_choice)
        ax.set_title(f"{metric_choice} per kelas")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        valid = per_class[per_class["support"] > 0][metric_choice]
        if len(valid) > 0:
            analisis_box(
                f"Untuk {metric_choice}, kelas terbaik adalah '{valid.idxmax()}' "
                f"({valid.max():.2f}) dan terendah '{valid.idxmin()}' ({valid.min():.2f}). "
                "Kelas dengan nilai rendah perlu perhatian: tambah data, perkuat augmentasi, "
                "atau periksa kemiripan visual dengan kelas lain pada confusion matrix."
            )
    else:
        st.warning("File `per_class.csv` tidak ditemukan.")

# ==========================================
# TAB 5: CONFUSION MATRIX
# ==========================================

with tabs[4]:
    st.header("Confusion Matrix")

    if cm is not None:
        names = class_names[: cm.shape[0]]
        mode = st.radio("Tampilan:", ["Jumlah", "Persentase per baris"], horizontal=True)

        if mode == "Jumlah":
            data, fmt, cmap, vmax = cm, "{:d}", "Blues", None
        else:
            row_sum = cm.sum(axis=1, keepdims=True)
            data = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
            fmt, cmap, vmax = "{:.2f}", "Greens", 1.0

        fig, ax = plt.subplots(figsize=(8.5, 7))
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label sebenarnya")
        thresh = data.max() / 2 if data.max() > 0 else 0.5
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                text = fmt.format(int(val)) if mode == "Jumlah" else fmt.format(val)
                ax.text(
                    j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        analisis_box(
            analysis.get("confusion_matrix")
            or "Diagonal utama menunjukkan prediksi benar; sel di luar diagonal adalah kelas yang "
            "tertukar."
        )

        # Pasangan kelas yang paling sering tertukar
        off = cm.copy().astype(int)
        np.fill_diagonal(off, 0)
        if off.sum() > 0:
            pairs = [
                (names[i], names[j], int(off[i, j]))
                for i in range(off.shape[0])
                for j in range(off.shape[1])
                if off[i, j] > 0
            ]
            pairs.sort(key=lambda x: x[2], reverse=True)
            st.subheader("Kesalahan yang paling sering terjadi")
            st.dataframe(
                pd.DataFrame(pairs[:8], columns=["Label sebenarnya", "Diprediksi sebagai", "Jumlah"]),
                hide_index=True,
                use_container_width=True,
            )
            top = pairs[0]
            analisis_box(
                f"Kesalahan paling sering: '{top[0]}' diprediksi sebagai '{top[1]}' ({top[2]} gambar). "
                "Pasangan kelas seperti ini biasanya mirip secara bentuk atau warna, sehingga "
                "penambahan data dan augmentasi pada kedua kelas dapat membantu."
            )
    else:
        st.warning("File `confusion_matrix.npy` tidak ditemukan.")

    with st.expander("Gambar confusion matrix dari notebook"):
        show_figure_file("6_confusion_matrix.png", "Confusion matrix (dari notebook)")

    st.subheader("Contoh hasil prediksi pada data uji")
    show_figure_file("7_sample_predictions.png", "Contoh prediksi pada data uji")
    analisis_box(
        analysis.get("prediksi")
        or "Judul hijau menandakan prediksi benar dan merah menandakan prediksi salah; "
        "persentase adalah confidence model."
    )

    st.subheader("Grad-CAM")
    show_figure_file("8_gradcam.png", "Visualisasi Grad-CAM")
    analisis_box(
        "Grad-CAM menyorot area gambar yang paling memengaruhi keputusan model (makin merah/hangat, "
        "makin berpengaruh). Jika area sorotan berada pada objek biota, model belajar dari fitur yang "
        "benar. Jika sorotan jatuh ke latar belakang (air atau dasar laut), prediksi bisa jadi "
        "bergantung pada konteks, bukan objek, dan berisiko salah pada gambar baru."
    )

# ==========================================
# TAB 6: PREDIKSI GAMBAR BARU (REAL-TIME)
# ==========================================

with tabs[5]:
    st.header("Uji Prediksi Gambar Baru")
    st.write(
        "Unggah gambar biota laut atau ambil foto lewat kamera. "
        "Prediksi ditampilkan langsung setelah gambar dimasukkan."
    )

    model, model_path = load_model()
    if model is None:
        st.error(
            "File model tidak ditemukan. Pastikan salah satu file ini ada:\n\n"
            + "\n".join(f"- `{p}`" for p in MODEL_CANDIDATES)
        )
    else:
        st.caption(f"Model: `{os.path.basename(model_path)}`")

        top_k = st.slider("Jumlah prediksi teratas", 1, len(class_names), 3)
        source = st.radio("Sumber gambar", ["Unggah file", "Kamera"], horizontal=True)

        if source == "Unggah file":
            file = st.file_uploader(
                "Unggah gambar", type=["jpg", "jpeg", "png", "bmp", "gif", "webp"]
            )
        else:
            file = st.camera_input("Ambil foto")

        if file is not None:
            try:
                image = Image.open(file)
            except Exception as e:
                st.error(f"Gambar tidak dapat dibaca: {e}")
                st.stop()

            def preprocess(img):
                import tensorflow as tf

                # Sama seperti training: RGB -> resize -> piksel 0-255.
                # Normalisasi [-1, 1] sudah dilakukan layer Rescaling di dalam model.
                img = ImageOps.exif_transpose(img).convert("RGB")
                arr = np.asarray(img, dtype=np.float32)
                arr = tf.image.resize(arr, IMG_SIZE).numpy()
                arr = np.round(arr).astype(np.float32)
                return np.expand_dims(arr, axis=0)

            with st.spinner("Memprediksi..."):
                probs = model.predict(preprocess(image), verbose=0)[0]

            order = np.argsort(probs)[::-1]
            best = int(order[0])
            names = class_names

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Gambar yang diuji", use_container_width=True)
            with col2:
                st.subheader("Hasil prediksi")
                st.metric(
                    "Kelas terprediksi",
                    names[best],
                    delta=f"{probs[best] * 100:.2f}% keyakinan",
                    delta_color="off",
                )
                if probs[best] < 0.5:
                    st.warning(
                        "Keyakinan model rendah. Gambar mungkin bukan biota laut, objeknya "
                        "kurang jelas, atau mirip dengan beberapa kelas sekaligus."
                    )
                for idx in order[:top_k]:
                    st.write(f"**{names[idx]}** — {probs[idx] * 100:.2f}%")
                    st.progress(float(probs[idx]))

            st.subheader("Probabilitas semua kelas")
            fig, ax = plt.subplots(figsize=(9, 4.5))
            colors = ["#E45756" if i == best else "#4C78A8" for i in range(len(names))]
            ax.barh(names, probs * 100, color=colors, edgecolor="black")
            ax.invert_yaxis()
            ax.set_xlabel("Probabilitas (%)")
            ax.set_xlim(0, 100)
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            second = int(order[1]) if len(order) > 1 else None
            msg = (
                f"Model memprediksi gambar ini sebagai '{names[best]}' dengan keyakinan "
                f"{probs[best]:.1%}."
            )
            if second is not None:
                msg += (
                    f" Kandidat berikutnya adalah '{names[second]}' ({probs[second]:.1%}); "
                    f"selisih keduanya {(probs[best] - probs[second]) * 100:.1f} poin persentase"
                )
                msg += (
                    ", sehingga prediksi cukup meyakinkan."
                    if probs[best] - probs[second] > 0.3
                    else ", sehingga model ragu antara kedua kelas ini."
                )
            analisis_box(msg)
        else:
            st.info("Belum ada gambar. Unggah file atau ambil foto untuk memulai.")