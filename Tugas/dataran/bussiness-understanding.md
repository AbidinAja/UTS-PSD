---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 1. Business Understanding

**Latar Belakang**  
Informasi mengenai tutupan dan penggunaan lahan sangat penting dalam perencanaan wilayah, manajemen sumber daya alam, hingga memantau perubahan iklim. Pendataan lahan secara manual dengan terjun langsung ke lapangan memakan waktu dan biaya operasional yang sangat besar. Untungnya, satelit observasi bumi seperti **Sentinel-2A** mengorbit dan mengambil citra bumi secara berkala. Namun, membedakan mana Sawah, Danau, atau Hutan dari citra satelit mentah tetap memerlukan pengamatan khusus dari tenaga ahli.

**Tujuan Bisnis**  
Mengaplikasikan teknik *Machine Learning* untuk mengklasifikasikan wilayah-wilayah di citra satelit ke dalam 6 kelas utama (Sawah, Bangunan, Mangrove, Hutan, Laut, dan Danau) secara otomatis, guna mempercepat efisiensi waktu pemetaan lahan. Kita juga akan membandingkan metode **Naive Bayes** dengan **K-Nearest Neighbors (KNN)** untuk mengetahui algoritma mana yang berkinerja lebih akurat.
