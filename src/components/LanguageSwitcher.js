import React, {useEffect, useState} from 'react';
import NavbarItem from '@theme/NavbarItem';
import DropdownNavbarItem from '@theme/NavbarItem/DropdownNavbarItem';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { translate } from '@docusaurus/Translate';

const LANG_EN = 'en';
const LANG_UR = 'ur';

const LanguageSwitcher = (props) => {
  const [currentLang, setCurrentLang] = useState(LANG_EN);
  const [isRTL, setIsRTL] = useState(false);

  useEffect(() => {
    // Check the current language and set RTL accordingly
    const htmlElement = document.documentElement;
    if (currentLang === LANG_UR) {
      htmlElement.setAttribute('dir', 'rtl');
      setIsRTL(true);
    } else {
      htmlElement.setAttribute('dir', 'ltr');
      setIsRTL(false);
    }
    
    // Store language preference in localStorage
    localStorage.setItem('preferred-language', currentLang);
  }, [currentLang]);

  const switchLanguage = (lang) => {
    setCurrentLang(lang);
  };

  const items = [
    {
      label: 'English',
      onClick: () => switchLanguage(LANG_EN),
      active: currentLang === LANG_EN,
    },
    {
      label: 'اردو',
      onClick: () => switchLanguage(LANG_UR),
      active: currentLang === LANG_UR,
    },
  ];

  // Add RTL class to body when Urdu is selected
  useEffect(() => {
    if (isRTL) {
      document.body.classList.add('rtl-layout');
    } else {
      document.body.classList.remove('rtl-layout');
    }
  }, [isRTL]);

  return (
    <DropdownNavbarItem
      {...props}
      label={currentLang === LANG_EN ? 'English' : 'اردو'}
      items={items}
    />
  );
};

export default LanguageSwitcher;