import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    CommonModule,
    RouterLink,
    MatCardModule,
    MatButtonModule,
    MatIconModule
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})
export class HomeComponent {
  features = [
    {
      title: 'گفتگوی هوشمند',
      description: 'با دستیار هوش مصنوعی ما تعامل کنید و پاسخ‌های دقیق دریافت کنید',
      icon: 'chat',
      route: '/chat',
      buttonText: 'شروع گفتگو'
    },
    {
      title: 'تنظیمات',
      description: 'پایگاه دانش را مدیریت کنید، اسناد را پردازش کنید و وضعیت سیستم را مشاهده کنید',
      icon: 'settings',
      route: '/settings',
      buttonText: 'مدیریت تنظیمات'
    },
    {
      title: 'مشارکت داده',
      description: 'دانش خود را به پایگاه داده ما اضافه کنید تا به بهبود پایگاه دانش پرشین وی کمک کنید',
      icon: 'add_circle',
      route: '/contribute',
      buttonText: 'مشارکت کنید'
    }
  ];

  systemFeatures = [
    {
      title: 'پردازش اسناد PDF',
      description: 'استخراج خودکار متن از فایل‌های PDF موجود در پوشه docs و تبدیل آن‌ها به بردارهای معنایی',
      icon: 'picture_as_pdf',
      color: '#e74c3c'
    },
    {
      title: 'پایگاه داده بردار',
      description: 'ذخیره‌سازی بردارها در ChromaDB برای بازیابی کارآمد و جستجوی معنایی پیشرفته',
      icon: 'storage',
      color: '#3498db'
    },
    {
      title: 'سیستم RAG',
      description: 'پیاده‌سازی سیستم تولید تقویت‌شده بازیابی با استفاده از LangChain و OpenAI',
      icon: 'psychology',
      color: '#9b59b6'
    },
    {
      title: 'پشتیبانی زبان فارسی',
      description: 'طراحی شده برای کار با متن فارسی در تمام مراحل پردازش و پاسخ‌دهی',
      icon: 'language',
      color: '#2ecc71'
    },
    {
      title: 'ارجاع به انسان',
      description: 'تشخیص خودکار زمانی که AI نمی‌تواند پاسخ مناسب ارائه دهد و ارجاع به کارشناسان انسانی',
      icon: 'support_agent',
      color: '#f39c12'
    },
    {
      title: 'API یکپارچه',
      description: 'ادغام یکپارچه با ساختار FastAPI موجود و ارائه endpoint های RESTful',
      icon: 'code',
      color: '#1abc9c'
    }
  ];

  technicalSpecs = [
    {
      title: 'OpenAI Embeddings',
      description: 'استفاده از مدل‌های embedding پیشرفته OpenAI برای تبدیل متن به بردار'
    },
    {
      title: 'ChromaDB Vector Store',
      description: 'پایگاه داده بردار با کارایی بالا برای ذخیره‌سازی و جستجوی معنایی'
    },
    {
      title: 'LangChain Framework',
      description: 'استفاده از فریمورک LangChain برای پیاده‌سازی سیستم RAG'
    },
    {
      title: 'Confidence Scoring',
      description: 'سیستم امتیازدهی اعتماد برای تشخیص نیاز به ارجاع انسانی'
    }
  ];

  apiEndpoints = [
    {
      method: 'POST',
      path: '/api/knowledge/query',
      description: 'جستجو در پایگاه دانش با سوال'
    },
    {
      method: 'POST',
      path: '/api/knowledge/process',
      description: 'پردازش تمام اسناد PDF در پوشه docs'
    },
    {
      method: 'GET',
      path: '/api/knowledge/status',
      description: 'بررسی وضعیت پردازش اسناد'
    }
  ];
}