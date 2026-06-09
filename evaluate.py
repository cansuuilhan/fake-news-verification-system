import pandas as pd
from app.services.retrieval_service import RetrievalService
from app.services.verification_service import VerificationService

def run_evaluation():
    print("="*60)
    print("🚀 FAKESENSE AKADEMİK MODEL DEĞERLENDİRME TESTİ")
    print("="*60)
    
    TEST_EDECEK_HABER_SAYISI = 100

    retrieval = RetrievalService()
    verification = VerificationService()
    
    dataset_path = "data/fake_real_tr/turkish_fake_real.csv"
    print(f"📦 {dataset_path} üzerinden veriler yükleniyor...")
    
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"❌ Veri seti okunurken hata oluştu: {str(e)}")
        return

    # --- NOKTA ATIŞI SÜTUN EŞLEŞTİRME ---
    # Terminal çıktısından aldığımız gerçek sütun isimlerini buraya sabitliyoruz
    text_col = 'clean_data'
    label_col = 'label'
    
    print(f"ℹ️ Okunan Haber Sütunu: '{text_col}' | Okunan Etiket Sütunu: '{label_col}'")
    # ------------------------------------

    sample_size = min(TEST_EDECEK_HABER_SAYISI, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    correct_predictions = 0
    total_tested = 0
    
    print(f"🔍 {sample_size} rastgele haber metni üzerinde toplu test başladı...\n")
    
    for idx, row in df_sample.iterrows():
        # Belirlediğimiz gerçek sütun isimlerinden veriyi çekiyoruz
        search_query = str(row[text_col]).strip()
        actual_label = str(row[label_col]).strip().lower()
        
        if len(search_query) < 15 or search_query.lower() == 'nan':
            continue
            
        search_query_truncated = search_query[:150]
        
        retrieved_docs = retrieval.retrieve(search_query_truncated, top_k=3)
        result = verification.verify_claim(search_query_truncated, retrieved_docs)
        predicted_verdict = result['verdict']
        
        is_correct = False
        
        # Gerçek Haber Kontrolü (1, true, gercek)
        if actual_label in ['1', 'true', 'gerçek', 'gercek', 'real']:
            if predicted_verdict in ['Destekleniyor', 'Kısmen Destekleniyor']:
                is_correct = True
                
        # Sahte Haber Kontrolü (0, fake, sahte)
        elif actual_label in ['0', 'fake', 'sahte']:
            if predicted_verdict in ['Kanıt Bulunamadı / Alakasız', 'Yetersiz Kanıt']:
                is_correct = True
            
        if is_correct:
            correct_predictions += 1
        total_tested += 1
        
        label_text = "GERÇEK" if actual_label in ['1', 'true', 'gerçek', 'gercek', 'real'] else "SAHTE"
        print(f"⏳ [{total_tested}/{sample_size}] | Haber Türü: {label_text} | Sistem Kararı: {predicted_verdict} -> {'✅ BAŞARILI' if is_correct else '❌ YANLIŞ'}")

    if total_tested == 0:
        print(f"\n❌ HATA: Satırlar süzülemedi, lütfen veri içeriğini kontrol edin.")
        return

    accuracy_rate = (correct_predictions / total_tested) * 100
    
    print("\n" + "="*60)
    print("📊 JÜRİ VE RAPOR İÇİN AKADEMİK PERFORMANS ÇIKTISI")
    print("="*60)
    print(f"🔹 Toplam Test Edilen Haber Sayısı : {total_tested}")
    print(f"🔹 Doğru Tahmin Edilen             : {correct_predictions}")
    print(f"🔹 Yanlış Tahmin Edilen            : {total_tested - correct_predictions}")
    print(f"🔥 SİSTEM GENEL ACCURACY (DOĞRULUK) : %{accuracy_rate:.2f}")
    print("="*60)

if __name__ == "__main__":
    run_evaluation()